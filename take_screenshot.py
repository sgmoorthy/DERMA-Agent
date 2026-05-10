import asyncio
from playwright.async_api import async_playwright

async def capture_screenshot():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.goto("http://localhost:8501")
        
        # Wait for streamlit app to load
        await page.wait_for_selector(".stApp")
        await page.wait_for_timeout(3000)  # Wait an extra 3 seconds to let everything render
        
        # Click the Generate Saliency Heatmap button
        buttons = await page.locator("button").all()
        for button in buttons:
            text = await button.inner_text()
            if "Generate Saliency Heatmap" in text:
                await button.click()
                break
                
        await page.wait_for_timeout(3000)
        
        for button in buttons:
            text = await button.inner_text()
            if "Render Latest Validation Plot" in text:
                await button.click()
                break
                
        await page.wait_for_timeout(3000)
        
        # Take the screenshot
        await page.screenshot(path="assets/app_demo.png", full_page=True)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(capture_screenshot())
