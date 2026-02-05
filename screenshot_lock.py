import threading

# Global lock used to serialize all calls to `pyautogui.screenshot` across threads.
# On some Windows setups/drivers, concurrent screen capture can hang or return invalid frames.
SCREENSHOT_LOCK = threading.Lock()

