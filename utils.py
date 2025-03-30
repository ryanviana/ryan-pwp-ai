import asyncio
import datetime
import logging
import logging.handlers  # For RotatingFileHandler
import os
import json
import re
import time
import random  # For additional random delays
from typing import List, Dict, Any, Optional

import pytz
from dateutil.parser import parse as dateutil_parse
from dateutil.relativedelta import relativedelta  # Now enabled

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from pydantic import BaseModel, ValidationError
from slugify import slugify as pyslugify
import tiktoken

# Selenium Imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    ElementClickInterceptedException,
    StaleElementReferenceException,  # Added
    WebDriverException,  # Added generic WebDriver exception
    NoSuchWindowException,
)
from webdriver_manager.chrome import ChromeDriverManager

# Selenium Stealth Import
from selenium_stealth import stealth

# --- Environment & Configuration ---
load_dotenv()
LINKEDIN_EMAIL = os.getenv("LINKEDIN_EMAIL")
LINKEDIN_PASSWORD = os.getenv("LINKEDIN_PASSWORD")
LINKEDIN_PROFILE_URL = os.getenv(
    "LINKEDIN_PROFILE_URL"
)  # Still needed to construct activity URL, e.g., "https://www.linkedin.com/in/your-profile-name/"
STATE_FILE = "last_timestamp.txt"
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)
LOG_FILE = "linkedin_agent.log"  # Log file for application specific logs

# Optional flags
MANUAL_MODE = (
    os.getenv("MANUAL_MODE", "false").lower() == "true"
)  # Set to "true" to disable headless and allow manual login
PROXY = os.getenv("PROXY")  # Optional proxy (e.g., "http://your-proxy:port")

# --- Logging Setup ---
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
log_handler.setFormatter(log_formatter)
root_logger = logging.getLogger()
if not root_logger.hasHandlers():
    root_logger.setLevel(LOG_LEVEL)
    root_logger.addHandler(log_handler)
logger = logging.getLogger(__name__)
logging.getLogger("selenium").setLevel(logging.WARNING)
logging.getLogger("webdriver_manager").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.INFO)


# --- Text Processing & Formatting ---
def clean_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    try:
        soup = BeautifulSoup(raw_html, "lxml")
        for tag in soup.find_all(
            ["p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"]
        ):
            tag.append("\n")
        text = soup.get_text(separator="", strip=True)
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error cleaning HTML: {e}")
        plain_soup = BeautifulSoup(raw_html, "lxml")
        return plain_soup.get_text(separator="\n", strip=True)


def estimate_reading_time(text: str, wpm=200) -> str:
    if not text:
        return "1 min read"
    try:
        word_count = len(re.findall(r"\w+", text))
        minutes = round(word_count / wpm)
        return f"{max(1, minutes)} min read"
    except Exception:
        logger.warning("Could not estimate reading time, defaulting to '1 min read'.")
        return "1 min read"


def generate_slug(text: str) -> str:
    if not text:
        return f"post-{int(datetime.datetime.now().timestamp())}"
    return pyslugify(text, max_length=80, save_order=True)


def get_current_date_iso() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d")


# --- Pydantic Validation Helper ---
def validate_data(
    data: dict, model_class: type[BaseModel]
) -> (Optional[BaseModel], Optional[str]):
    try:
        validated_obj = model_class(**data)
        logger.debug(f"Validation successful for {model_class.__name__}")
        return validated_obj, None
    except Exception as e:
        error_msg = f"Validation error for {model_class.__name__}: {e}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg


# --- Timestamp Handling for State ---
def load_last_timestamp() -> datetime.datetime:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            timestamp_str = f.read().strip()
            if not timestamp_str:
                raise ValueError("Timestamp file is empty.")
            dt = dateutil_parse(timestamp_str)
            dt_aware = (
                pytz.utc.localize(dt) if dt.tzinfo is None else dt.astimezone(pytz.utc)
            )
            logger.info(f"Loaded last processed timestamp: {dt_aware.isoformat()}")
            return dt_aware
    except FileNotFoundError:
        logger.warning(
            f"'{STATE_FILE}' not found. Will process posts from start of today (UTC)."
        )
        start_of_today_utc = datetime.datetime.now(pytz.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        save_last_timestamp(start_of_today_utc)
        return start_of_today_utc
    except Exception as e:
        logger.error(
            f"Error loading timestamp: {e}. Defaulting to start of today (UTC)."
        )
        start_of_today_utc = datetime.datetime.now(pytz.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return start_of_today_utc


def save_last_timestamp(timestamp: datetime.datetime):
    if not timestamp:
        logger.error("Attempted to save an invalid timestamp.")
        return
    try:
        ts_aware = (
            pytz.utc.localize(timestamp)
            if timestamp.tzinfo is None
            else timestamp.astimezone(pytz.utc)
        )
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            f.write(ts_aware.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        logger.info(f"Saved last processed timestamp: {ts_aware.isoformat()}")
    except Exception as e:
        logger.error(f"Error saving timestamp: {e}", exc_info=True)


# --- Timestamp Parser (Relative Time) ---
def parse_linkedin_timestamp(time_str: str) -> Optional[datetime.datetime]:
    if not time_str:
        return None
    # Remove any trailing "• ..." and extra spaces
    time_str = re.sub(r"\s*•.*$", "", time_str).strip()
    time_str = time_str.replace("Liked", "").strip()
    now = datetime.datetime.now(pytz.utc)
    try:
        # Try matching minutes, hours, days, weeks
        m = re.match(r"^(\d+)\s*m$", time_str)
        h = re.match(r"^(\d+)\s*h$", time_str)
        d = re.match(r"^(\d+)\s*d$", time_str)
        w = re.match(r"^(\d+)\s*w$", time_str)
        if m:
            return now - datetime.timedelta(minutes=int(m.group(1)))
        if h:
            return now - datetime.timedelta(hours=int(h.group(1)))
        if d:
            return now - datetime.timedelta(days=int(d.group(1)))
        if w:
            return now - datetime.timedelta(weeks=int(w.group(1)))
        # Handle months and years using relativedelta
        mth = re.match(r"^(\d+)\s*mo", time_str)
        yr = re.match(r"^(\d+)\s*yr", time_str)
        if mth:
            return now - relativedelta(months=int(mth.group(1)))
        if yr:
            return now - relativedelta(years=int(yr.group(1)))
        # Fallback to dateutil_parse for absolute dates
        parsed_date = dateutil_parse(time_str)
        if not hasattr(parsed_date, "year") or parsed_date.year == now.year:
            temp_date_this_year = parsed_date.replace(year=now.year)
            temp_naive_utc = (
                pytz.utc.localize(temp_date_this_year)
                if temp_date_this_year.tzinfo is None
                else temp_date_this_year.astimezone(pytz.utc)
            )
            if temp_naive_utc > now:
                parsed_date = parsed_date.replace(year=now.year - 1)
            else:
                parsed_date = parsed_date.replace(year=now.year)
        parsed_date_aware = (
            pytz.utc.localize(parsed_date)
            if parsed_date.tzinfo is None
            else parsed_date.astimezone(pytz.utc)
        )
        return parsed_date_aware
    except Exception as e:
        logger.warning(f"Could not parse timestamp '{time_str}': {e}")
        return None


# --- LinkedIn Monitor Implementation (Using Selenium with Persistent Profile) ---
def run_selenium_monitor_sync() -> List[Dict[str, Any]]:
    logger.info("--- Starting Selenium Monitor Sync Task ---")
    if not all([LINKEDIN_EMAIL, LINKEDIN_PASSWORD, LINKEDIN_PROFILE_URL]):
        logger.critical("Missing LinkedIn credentials or profile URL in .env file.")
        return []

    last_timestamp_dt = load_last_timestamp()
    logger.info(f"Fetching posts newer than: {last_timestamp_dt.isoformat()}")
    new_posts_data = []
    driver = None
    max_scrolls = 15
    scroll_pause_time = 3.5
    element_timeout = 15

    # --- Selenium Selectors ---
    login_username_id = "username"
    login_password_id = "password"
    login_submit_button_xpath = "//button[@type='submit']"
    # For login detection we use our profile's recent activity page post container.
    post_container_xpath = (
        "//div[contains(@class, 'feed-shared-update-v2') and @data-urn]"
    )
    timestamp_xpath = ".//span[contains(@class, 'update-components-actor__sub-description')]//span[@aria-hidden='true']"
    content_wrapper_xpath = ".//div[contains(@class, 'feed-shared-update-v2__description-wrapper')] | .//div[contains(@class, 'update-components-text')]"
    text_content_css_selector = "div.feed-shared-update-v2__description span[dir='ltr'], div.update-components-text span[dir='ltr'], div.feed-shared-inline-show-more-text"
    see_more_button_xpath = ".//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'more')]"
    image_xpath = ".//div[contains(@class, 'update-components-image__container')]//img | .//div[contains(@class, 'feed-shared-image__container')]//img"
    video_xpath = ".//div[contains(@class, 'update-components-linkedin-video__container')]//video | .//div[contains(@class, 'feed-shared-linkedin-video__container')]//video"
    article_link_xpath = ".//a[contains(@class, 'feed-shared-article__figure') or contains(@class, 'feed-shared-article__meta')]"

    # Define profile path for Chrome data persistence
    script_dir = os.path.dirname(os.path.abspath(__file__))
    profile_path = os.path.join(script_dir, "chrome_profile")
    logger.info(f"Using Chrome profile directory: {profile_path}")

    try:
        logger.info("Setting up Chrome WebDriver...")
        options = webdriver.ChromeOptions()

        # --- Headless Mode ---
        if not MANUAL_MODE:
            options.add_argument("--headless=new")
        else:
            logger.info(
                "Manual mode enabled. Running browser in visible mode for manual login."
            )

        # --- Proxy Support (if provided) ---
        if PROXY:
            options.add_argument(f"--proxy-server={PROXY}")
            logger.info(f"Using proxy: {PROXY}")

        # Common options
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(
            "user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
        )
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option(
            "prefs",
            {
                "credentials_enable_service": False,
                "profile.password_manager_enabled": False,
            },
        )
        options.add_argument(f"--user-data-dir={profile_path}")
        options.add_argument("--profile-directory=Default")

        logger.info("Initializing WebDriver Service...")
        service = ChromeService(ChromeDriverManager().install())
        logger.info("Initializing Chrome Driver...")
        driver = webdriver.Chrome(service=service, options=options)
        driver.implicitly_wait(3)
        logger.info("WebDriver initialized.")

        # --- Apply Selenium Stealth ---
        stealth(
            driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
        )
        logger.info("Applied selenium-stealth measures.")

        # --- Login Attempt ---
        # Navigate directly to the activity page for our profile.
        activity_url = f"{LINKEDIN_PROFILE_URL.rstrip('/')}/recent-activity/all/"
        driver.get(activity_url)
        time.sleep(random.uniform(1.0, 2.0))
        is_logged_in = False
        current_url = driver.current_url
        if (
            "login" in current_url
            or "authwall" in current_url
            or "checkpoint" in current_url
        ):
            logger.info("Not logged in; login required.")
        else:
            try:
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, post_container_xpath))
                )
                is_logged_in = True
                logger.info("Detected active session; recent activity posts present.")
            except TimeoutException:
                logger.info(
                    "Recent activity posts not detected; assuming not logged in."
                )

        if not is_logged_in:
            if MANUAL_MODE:
                logger.info(
                    "Manual login mode: Please log in manually in the opened browser window (solve CAPTCHA if prompted), and then navigate to your profile's recent activity page."
                )
                input(
                    "After completing manual login and ensuring your recent activity page is loaded, press Enter to continue..."
                )
                driver.get(activity_url)
                time.sleep(5)
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, post_container_xpath))
                    )
                    is_logged_in = True
                    logger.info(
                        "Manual login successful; recent activity posts detected."
                    )
                except TimeoutException:
                    logger.error(
                        "Manual login failed: Recent activity posts not detected."
                    )
                    raise ConnectionRefusedError("Manual login failed.")
            else:
                logger.info("Not logged in. Attempting automatic login sequence.")
                driver.get("https://www.linkedin.com/login")
                try:
                    user_field = WebDriverWait(driver, element_timeout).until(
                        EC.visibility_of_element_located((By.ID, login_username_id))
                    )
                    pass_field = driver.find_element(By.ID, login_password_id)
                    logger.info(f"Entering credentials for {LINKEDIN_EMAIL}...")
                    user_field.send_keys(LINKEDIN_EMAIL)
                    time.sleep(random.uniform(0.6, 1.2))
                    pass_field.send_keys(LINKEDIN_PASSWORD)
                    time.sleep(random.uniform(1.0, 1.5))
                    driver.find_element(By.XPATH, login_submit_button_xpath).click()
                except Exception as login_field_err:
                    logger.error(
                        f"Error interacting with login fields: {login_field_err}"
                    )
                    driver.save_screenshot("selenium_login_field_error.png")
                    raise TimeoutError("Failed to interact with login fields.")
                try:
                    logger.info("Waiting for post-login state...")
                    driver.get(activity_url)
                    time.sleep(5)
                    WebDriverWait(driver, element_timeout + 10).until(
                        EC.presence_of_element_located((By.XPATH, post_container_xpath))
                    )
                    logger.info(
                        "Automatic login successful; recent activity posts detected."
                    )
                except TimeoutException:
                    logger.error(
                        "Automatic login failed: Recent activity posts not detected."
                    )
                    driver.save_screenshot("selenium_login_timeout_error.png")
                    raise ConnectionRefusedError("Automatic login failed.")
        else:
            logger.info("Using existing logged-in session.")

        # --- Navigate to Activity Feed ---
        logger.info(f"Navigated to activity feed: {activity_url}")
        try:
            WebDriverWait(driver, element_timeout).until(
                EC.presence_of_element_located((By.XPATH, post_container_xpath))
            )
            logger.info("Activity feed loaded.")
            time.sleep(3)
        except TimeoutException:
            logger.warning(
                "Activity feed loaded, but no post containers found initially."
            )

        # --- Scroll and Scrape ---
        logger.info("Scrolling down to load posts...")
        processed_urns = set()
        extracted_posts_dict = {}
        found_older_post = False
        current_max_timestamp = last_timestamp_dt
        consecutive_no_change_scrolls = 0

        for i in range(max_scrolls):
            scroll_start_time = time.monotonic()
            logger.debug(f"Scroll attempt {i+1}/{max_scrolls}")
            last_height = driver.execute_script("return document.body.scrollHeight")
            try:
                post_elements = driver.find_elements(By.XPATH, post_container_xpath)
                logger.debug(f"Found {len(post_elements)} post elements in DOM.")
                for post_element in post_elements:
                    post_urn = post_element.get_attribute("data-urn")
                    if not post_urn or post_urn in processed_urns:
                        continue
                    processed_urns.add(post_urn)
                    try:
                        time_element = post_element.find_element(
                            By.XPATH, timestamp_xpath
                        )
                        time_str = time_element.text
                        post_timestamp_dt = parse_linkedin_timestamp(time_str)
                    except NoSuchElementException:
                        logger.debug(
                            f"Timestamp element not found for post {post_urn}. Skipping."
                        )
                        continue
                    except Exception as ts_parse_err:
                        logger.warning(
                            f"Error parsing timestamp '{time_str}' for {post_urn}: {ts_parse_err}"
                        )
                        continue
                    if not post_timestamp_dt:
                        logger.warning(
                            f"Skipping post {post_urn} due to invalid/unparseable timestamp."
                        )
                        continue
                    if post_timestamp_dt <= last_timestamp_dt:
                        logger.debug(
                            f"Post {post_urn} ({post_timestamp_dt.isoformat()}) is not newer. Marking potential end."
                        )
                        found_older_post = True
                        continue
                    post_data = {"id": post_urn, "timestamp_dt": post_timestamp_dt}
                    post_text_extracted = False  # Flag to track if text was extracted
                    see_more_clicked_successfully = False
                    try:
                        see_more_button = post_element.find_element(
                            By.XPATH, see_more_button_xpath
                        )
                        if see_more_button.is_displayed():
                            driver.execute_script(
                                "arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});",
                                see_more_button,
                            )
                            time.sleep(random.uniform(0.3, 0.6))
                            driver.execute_script(
                                "arguments[0].click();", see_more_button
                            )
                            # Wait for the button to disappear or become stale
                            try:
                                WebDriverWait(driver, 5).until(
                                    EC.staleness_of(see_more_button)
                                )
                                logger.debug(
                                    f"Clicked 'See more' and waited for staleness for post {post_urn}"
                                )
                                see_more_clicked_successfully = True
                            except TimeoutException:
                                logger.warning(
                                    f"'See more' button did not become stale for post {post_urn} after click."
                                )
                    except NoSuchElementException:
                        # No 'See more' button found
                        pass
                    except (
                        ElementClickInterceptedException,
                        StaleElementReferenceException,
                    ) as click_ex:
                        logger.warning(
                            f"'See more' click failed for {post_urn}: {type(click_ex).__name__}."
                        )
                    except Exception as see_more_err:
                        logger.warning(
                            f"Unexpected error interacting with 'See more' for {post_urn}: {see_more_err}"
                        )

                    # --- Text Extraction ---
                    # Attempt to extract full text if 'See more' was clicked successfully
                    if see_more_clicked_successfully:
                        try:
                            # Re-find the specific post container using its URN
                            # This ensures we have a fresh reference after potential DOM changes
                            refreshed_post_element = driver.find_element(
                                By.XPATH, f"//div[@data-urn='{post_urn}']"
                            )
                            wrapper_after_click = refreshed_post_element.find_element(
                                By.XPATH, content_wrapper_xpath
                            )
                            text_elements_after_click = (
                                wrapper_after_click.find_elements(
                                    By.CSS_SELECTOR, text_content_css_selector
                                )
                            )
                            post_text = "\n".join(
                                [el.text for el in text_elements_after_click if el.text]
                            ).strip()
                            if not post_text:
                                post_text = wrapper_after_click.text.strip()  # Fallback
                            post_data["text"] = post_text
                            post_text_extracted = True
                            logger.debug(
                                f"Extracted full text for {post_urn} after 'See more' click."
                            )
                        except Exception as text_extract_err:
                            logger.error(
                                f"Error extracting text for {post_urn} *after* 'See more' click (using refreshed element): {text_extract_err}"
                            )

                    # Fallback: Extract preview text if full text wasn't extracted
                    if not post_text_extracted:
                        try:
                            # Use the original post_element reference here
                            wrapper = post_element.find_element(
                                By.XPATH, content_wrapper_xpath
                            )
                            text_elements = wrapper.find_elements(
                                By.CSS_SELECTOR, text_content_css_selector
                            )
                            post_text = "\n".join(
                                [el.text for el in text_elements if el.text]
                            ).strip()
                            if not post_text:
                                post_text = wrapper.text.strip()
                            post_data["text"] = post_text
                            if not post_text:
                                logger.warning(
                                    f"Preview text for {post_urn} also seems empty."
                                )
                            else:
                                logger.debug(f"Extracted preview text for {post_urn}.")
                        except StaleElementReferenceException:
                            logger.warning(
                                f"Original post element became stale while trying to extract preview text for {post_urn}. Setting empty text."
                            )
                            post_data["text"] = ""
                        except NoSuchElementException:
                            logger.warning(
                                f"Text content wrapper not found for post {post_urn}. Setting empty text."
                            )
                            post_data["text"] = ""
                        except Exception as text_err:
                            logger.error(
                                f"Error extracting preview text for {post_urn}: {text_err}"
                            )
                            post_data["text"] = ""
                    # ------------------------------------------------------------------------

                    media_url = None
                    try:
                        img_element = post_element.find_element(By.XPATH, image_xpath)
                        media_url = img_element.get_attribute("src")
                    except NoSuchElementException:
                        try:
                            video_element = post_element.find_element(
                                By.XPATH, video_xpath
                            )
                            media_url = video_element.get_attribute(
                                "src"
                            ) or video_element.find_element(
                                By.TAG_NAME, "source"
                            ).get_attribute(
                                "src"
                            )
                        except NoSuchElementException:
                            try:
                                article_element = post_element.find_element(
                                    By.XPATH, article_link_xpath
                                )
                                media_url = article_element.get_attribute("href")
                            except NoSuchElementException:
                                pass
                    except Exception as media_err:
                        logger.warning(
                            f"Error extracting media/link for {post_urn}: {media_err}"
                        )
                    post_data["media_url"] = media_url
                    if post_data.get("text"):
                        extracted_posts_dict[post_urn] = post_data
                        logger.debug(
                            f"Stored post {post_urn} with timestamp {post_timestamp_dt.isoformat()}."
                        )
                        if post_timestamp_dt > current_max_timestamp:
                            current_max_timestamp = post_timestamp_dt
                    else:
                        logger.warning(
                            f"Skipping post {post_urn} due to empty text content."
                        )
            except Exception as outer_loop_err:
                logger.error(
                    f"Error during scroll iteration {i}: {outer_loop_err}",
                    exc_info=True,
                )
            if found_older_post:
                logger.info("Found post older than cutoff timestamp. Stopping scroll.")
                break
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            scroll_wait_start = time.monotonic()
            while time.monotonic() < scroll_wait_start + scroll_pause_time:
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height > last_height + 50:
                    logger.debug("New content loaded after scroll.")
                    break
                time.sleep(0.5)
            else:
                logger.info("Scroll height did not change significantly.")
                consecutive_no_change_scrolls += 1
                if consecutive_no_change_scrolls >= 2:
                    logger.info(
                        "No change in scroll height for 2 consecutive attempts. Ending scroll."
                    )
                    break
                else:
                    logger.debug(
                        f"Consecutive no-change scrolls: {consecutive_no_change_scrolls}"
                    )
            if (
                driver.execute_script("return document.body.scrollHeight")
                > last_height + 50
            ):
                consecutive_no_change_scrolls = 0
            logger.debug(
                f"Scroll attempt {i+1} took {time.monotonic() - scroll_start_time:.2f}s"
            )

        logger.info(
            f"Finished scrolling. Processed {len(extracted_posts_dict)} new posts."
        )
        for post_urn, post_data in extracted_posts_dict.items():
            new_posts_data.append(
                {
                    "id": post_urn,
                    "text": post_data["text"],
                    "media_url": post_data.get("media_url"),
                    "timestamp": post_data["timestamp_dt"].isoformat(),
                }
            )
        new_posts_data.sort(key=lambda p: p["timestamp"])
        if new_posts_data:
            logger.info(f"Identified {len(new_posts_data)} new posts.")
            if current_max_timestamp > last_timestamp_dt:
                save_last_timestamp(current_max_timestamp)
            else:
                logger.info(
                    "No posts found strictly newer than the last saved timestamp."
                )
        else:
            logger.info("No new posts found meeting the criteria.")

    except ConnectionRefusedError as e:
        logger.critical(f"LinkedIn Monitor failed during login/checkpoint: {e}")
        return []
    except WebDriverException as e:
        logger.critical(f"WebDriver error during monitoring: {e}", exc_info=True)
        if driver:
            try:
                driver.save_screenshot("selenium_webdriver_error.png")
            except Exception:
                pass
        return []
    except Exception as e:
        logger.critical(f"Unexpected error during monitoring: {e}", exc_info=True)
        if driver:
            try:
                driver.save_screenshot("selenium_runtime_error.png")
            except Exception:
                pass
        return []
    finally:
        if driver:
            logger.info("Closing WebDriver...")
            try:
                driver.quit()
            except Exception as quit_err:
                logger.error(f"Error closing WebDriver: {quit_err}")
    logger.info(
        f"--- Selenium Monitor Sync Task Finished. Returning {len(new_posts_data)} posts. ---"
    )
    return new_posts_data


async def linkedin_monitor() -> List[Dict[str, Any]]:
    loop = asyncio.get_running_loop()
    try:
        logger.debug("Submitting Selenium task to executor...")
        result = await loop.run_in_executor(None, run_selenium_monitor_sync)
        logger.debug("Selenium task completed.")
        return result
    except Exception as e:
        logger.critical(
            f"Error executing Selenium monitor in executor: {e}", exc_info=True
        )
        return []
