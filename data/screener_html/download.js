const fs = require("fs");
const path = require("path");

const tickers = `
JSLL
`.trim().split(/\s+/);

const MAX_AGE_DAYS = 7;
const REFRESH_AFTER = new Date(
    Date.now() - MAX_AGE_DAYS * 24 * 60 * 60 * 1000
);

        
// Delay between requests (milliseconds)
const DELAY = 15000;

// --------------------------------------------------------------------

function shouldDownload(ticker) {
    const file = path.join(__dirname, `${ticker}.html`);

    if (!fs.existsSync(file)) {
        return true;
    }

    const stats = fs.statSync(file);

    return stats.mtime < REFRESH_AFTER;
}

async function downloadTicker(ticker) {

    const url = `https://www.screener.in/company/${encodeURIComponent(ticker)}/`;

    try {

        const response = await fetch(url, {
            headers: {
                "User-Agent": "Mozilla/5.0"
            }
        });

        if (!response.ok) {
            throw new Error(`${response.status} ${response.statusText}`);
        }

        const html = await response.text();

        fs.writeFileSync(
            path.join(__dirname, `${ticker}.html`),
            html
        );

        console.log(`Downloaded ${ticker}`);

    } catch (err) {
        console.error(`Failed ${ticker}: ${err.message}`);
    }
}

(async () => {

    const pending = tickers.filter(shouldDownload);

    console.log(`Total tickers : ${tickers.length}`);
    console.log(`Need download : ${pending.length}`);
    console.log(`Already fresh : ${tickers.length - pending.length}`);
    console.log("");

    for (const ticker of pending) {

        await downloadTicker(ticker);

        await new Promise(r => setTimeout(r, DELAY));
    }

    console.log("Done.");

})();


        
