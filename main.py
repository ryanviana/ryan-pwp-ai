import asyncio
import logging
import json
from typing import List, Dict, Any, cast

from langgraph.graph import StateGraph, END, Graph

# Import schemas, agents, and utilities
from schemas import AgentState, ClassificationType
from agents import (
    triage_agent,
    transform_to_blog,
    transform_to_work_experience,
    transform_to_education,
    transform_to_achievement,
    transform_to_skill,
    content_publisher_agent,
)
from utils import logger, linkedin_monitor  # Use the real monitor

# --- Graph Definition ---

# Map classification types to the corresponding transformation node names in the graph
TRANSFORMATION_NODE_MAP = {
    "blog": "transform_blog",
    "work-experience": "transform_work_experience",
    "education": "transform_education",
    "achievement": "transform_achievement",
    "skill": "transform_skill",
}

# Define the graph state using the AgentState TypedDict
workflow = StateGraph(AgentState)

# --- Add Nodes ---
logger.info("Defining graph nodes...")
workflow.add_node("triage", triage_agent)
workflow.add_node("transform_blog", transform_to_blog)
workflow.add_node("transform_work_experience", transform_to_work_experience)
workflow.add_node("transform_education", transform_to_education)
workflow.add_node("transform_achievement", transform_to_achievement)
workflow.add_node("transform_skill", transform_to_skill)
workflow.add_node("publish", content_publisher_agent)
logger.info("Nodes added to graph.")

# --- Define Edges ---

# Entry point
workflow.set_entry_point("triage")
logger.info("Graph entry point set to 'triage'.")


# Conditional Edges after Triage
def decide_transformations(state: AgentState) -> List[str]:
    """Determines the next node(s) based on triage classifications."""
    classifications = state.get("classifications", [])
    logger.debug(f"Conditional Edge: Received classifications: {classifications}")
    next_nodes = []
    for classification in classifications:
        if classification in TRANSFORMATION_NODE_MAP:
            node_name = TRANSFORMATION_NODE_MAP[classification]
            next_nodes.append(node_name)
        else:
            logger.warning(
                f"Conditional Edge: No transformation node mapped for classification: '{classification}'"
            )

    if not next_nodes:
        logger.info(
            "Conditional Edge: No relevant classifications found. Routing directly to 'publish'."
        )
        # If no transformations are needed, go straight to publish (which will likely do nothing)
        # Alternatively, could route directly to END if publishing empty state is pointless.
        return ["publish"]
    else:
        logger.info(
            f"Conditional Edge: Routing to transformation node(s): {next_nodes}"
        )
        return next_nodes


# Add the conditional branching logic
# LangGraph handles the case where decide_transformations returns a list of nodes.
# The keys in the third argument map the *possible return values* (node names) to the *actual nodes* to execute.
node_mapping_for_conditional_edge = {
    node_name: node_name for node_name in TRANSFORMATION_NODE_MAP.values()
}
node_mapping_for_conditional_edge["publish"] = (
    "publish"  # Add the direct-to-publish route
)

workflow.add_conditional_edges(
    source="triage",
    path=decide_transformations,
    path_map=node_mapping_for_conditional_edge,
)
logger.info("Conditional edges defined from 'triage' based on classification.")


# Edges from Transformation Nodes to Publisher
# All transformation branches should eventually lead to the publish node.
# LangGraph ensures the 'publish' node runs only after all its dependencies (the triggered transform nodes) complete.
for node_name in TRANSFORMATION_NODE_MAP.values():
    workflow.add_edge(node_name, "publish")
logger.info("Edges defined from all transformation nodes to 'publish'.")

# Final Edge from Publisher to End
workflow.add_edge("publish", END)
logger.info("Edge defined from 'publish' to END.")

# --- Compile the Graph ---
try:
    app: Graph = workflow.compile()
    logger.info("LangGraph workflow compiled successfully.")
    # Optional: Visualize the graph if needed (requires graphviz)
    # try:
    #     app.get_graph().draw_mermaid_png(output_file_path="graph_mermaid.png")
    #     app.get_graph().draw_png(output_file_path="graph.png")
    #     logger.info("Graph visualizations saved to graph_mermaid.png and graph.png")
    # except Exception as viz_err:
    #     logger.warning(f"Could not generate graph visualizations: {viz_err}. Install graphviz and pygraphviz/pydot if needed.")

except Exception as compile_err:
    logger.critical(
        f"Failed to compile LangGraph workflow: {compile_err}", exc_info=True
    )
    raise

# --- Graph Execution Logic ---


async def process_post(post_data: Dict[str, Any], graph_app: Graph) -> Dict:
    """Runs the LangGraph workflow for a single LinkedIn post."""
    post_id = post_data.get("id", "N/A")
    logger.info(f"\n--- Processing Post ID: {post_id} ---")
    logger.debug(f"Raw Post Data: {json.dumps(post_data, indent=2)}")

    # Initialize the state for this post
    initial_state: AgentState = {
        "raw_post_data": post_data,
        "classifications": [],
        "transformed_data": {},
        "publish_results": {},
        "error_messages": [],
    }

    final_state = {}
    try:
        # Use astream_events for detailed logging (v2 API)
        # Or use ainvoke for just the final state
        # Using astream to get intermediate results if needed, but ainvoke is simpler for final state
        final_state = await graph_app.ainvoke(
            initial_state
        )  # , {"recursion_limit": 10}

        logger.info(f"--- Finished Processing Post ID: {post_id} ---")
        # Log summary from final state
        logger.info(f"  Classifications: {final_state.get('classifications')}")
        logger.info(
            f"  Transformed Types: {list(final_state.get('transformed_data', {}).keys())}"
        )
        logger.info(
            f"  Publish Results Count: {len(final_state.get('publish_results', {}))}"
        )
        if final_state.get("publish_results"):
            logger.debug(
                f"  Publish Details: {json.dumps(final_state.get('publish_results'), indent=2)}"
            )
        if final_state.get("error_messages"):
            logger.warning(f"  Errors Encountered: {final_state.get('error_messages')}")
        else:
            logger.info("  No errors reported.")
        logger.info("-------------------------------------------\n")

    except Exception as e:
        logger.critical(f"FATAL ERROR processing post {post_id}: {e}", exc_info=True)
        # Store minimal final state indicating failure
        final_state = {
            "raw_post_data": post_data,
            "error_messages": [f"Graph execution failed: {e}"]
            + initial_state.get("error_messages", []),
        }

    return final_state


async def main():
    """Main function to fetch posts and run the workflow."""
    logger.info("=============================================")
    logger.info("Starting AI Agent System for LinkedIn Content")
    logger.info("=============================================")

    # 1. Fetch new posts using the LinkedIn monitor
    logger.info("Running LinkedIn Monitor...")
    try:
        new_posts = await linkedin_monitor()
    except Exception as monitor_err:
        logger.critical(f"LinkedIn Monitor failed to run: {monitor_err}", exc_info=True)
        new_posts = []  # Ensure it's an empty list on failure

    if not new_posts:
        logger.info("No new posts detected by LinkedIn monitor. Exiting.")
        return

    logger.info(f"LinkedIn monitor returned {len(new_posts)} new post(s) to process.")

    # 2. Process each post sequentially using the compiled graph
    # Processing sequentially is safer for API rate limits and easier debugging
    all_results = []
    for post in new_posts:
        result = await process_post(post, app)
        all_results.append(result)
        # Optional: Small delay between processing posts
        # await asyncio.sleep(1)

    # 3. Final Summary (optional)
    logger.info("=============================================")
    logger.info("Finished processing all detected posts.")
    success_count = sum(1 for r in all_results if not r.get("error_messages"))
    error_count = len(all_results) - success_count
    logger.info(
        f"Summary: {success_count} post(s) processed successfully, {error_count} post(s) encountered errors."
    )
    logger.info("=============================================")


if __name__ == "__main__":
    # Setup asyncio event loop and run main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as main_err:
        logger.critical(
            f"Unhandled exception in main execution: {main_err}", exc_info=True
        )
