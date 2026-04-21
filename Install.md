How to install TruthLens

TruthLens is a browser extension that scans news artcles for hyperpartisan and manipulative language.

Before you start:
 - Chromium based browser: Google chrome, Microsft edge, Brave, others.
 - The source code: Make sure you have downloaded or cloned the TruthLens extension folder to your pc.
 - Backend server: As TruthLens talks with an LLM, make sure the backend is running on http://127.0.0.1:8000/ befor eusing the extension.

Installation steps (Brave browser):
1. Open the extensions page
    Launch the browser, type 'chrome://extensions/' into the search bar and enter
    OR 
    Launch the browser, find the 3 dots in the top right corner, click 'Extensions', then click 'Manage extension'.

2. Enable developer mode:
    In the top right of the extensions page you will see 'Developer mode', make sure its on.

3. Load the unpacked extension:
    Once developer mode is on, a new button will appear in the top left under 'Extension' it's called 'load unpacked', press it .

4. Select extension folder:
    A file explorer window will open. Find the TruthLens folder on your pc (this is hyperpartisan_extension) and press select folder.

5. Pin the extension:
    TruthLens will now appear in your list of installed extensions. To make it easily accessible you can go the jigsaw icon (🧩) in the top right corner, then press the pin icon next to TruthLens.


Using TruthLens:
1. Install the requirements, then start the local backend server 
2. Navgate to an news article you want to analyse
3. Press the TruthLens icon in your browser toolbar and press 'analyse article' to get a detailed analysis of that news article.






