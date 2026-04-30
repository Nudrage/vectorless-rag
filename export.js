import puppeteer from "puppeteer";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import { execSync } from "child_process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DELAY = (ms) => new Promise((r) => setTimeout(r, ms));
const BATCH_SIZE = 15;

async function exportPDF() {
  // ── Step 0 ──────────────────────────────────────────────────────────────
  // Convert OpenAPI spec to ReDoc HTML (equivalent to: redoc-cli bundle spec -o openapi.html)
  const specInput = process.argv[2];
  if (!specInput) {
    console.error("Usage: node export.js <path-to-openapi-spec>");
    process.exit(1);
  }
  const specPath = path.resolve(specInput);
  const specDir = path.dirname(specPath);
  const specBase = path.basename(specPath, path.extname(specPath));
  const htmlPath = path.join(specDir, `${specBase}.html`);
  const pdfPath = path.join(specDir, `${specBase}.pdf`);

  if (!fs.existsSync(specPath)) {
    console.error(`Error: OpenAPI spec file not found: ${specPath}`);
    process.exit(1);
  }

  console.log(`Step 0: Generating ReDoc HTML from ${specInput}...`);
  const redoclyCli = path.join(__dirname, "node_modules", ".bin", "redocly");
  try {
    execSync(`"${redoclyCli}" build-docs "${specPath}" -o "${htmlPath}"`, {
      cwd: __dirname,
      stdio: "inherit",
    });
  } catch (err) {
    console.error("Failed to generate HTML from OpenAPI spec:", err.message);
    process.exit(1);
  }
  console.log(`HTML generated: ${htmlPath}`);

  const browser = await puppeteer.launch({ headless: "new", protocolTimeout: 0 });
  const page = await browser.newPage();
  page.setDefaultTimeout(600_000);

  const filePath = `file://${htmlPath}`;

  await page.goto(filePath, { waitUntil: "networkidle0" });

  // Wait for ReDoc to fully load
  await page.waitForFunction(() => document.body.innerText.length > 1000);
  await DELAY(2000);

  // ── Step 1 ──────────────────────────────────────────────────────────────
  // Expand ALL response code dropdown buttons (every status code).
  //
  // ReDoc response buttons are ALL collapsed on initial load
  // (parentElement.children.length === 1). Clicking a button makes React
  // render a sibling <div> with the response schema details.
  //
  // We click in small batches to avoid overwhelming React.
  const responseButtons = await page.evaluate(() => {
    const buttons = [];
    document.querySelectorAll("button").forEach((btn, idx) => {
      const strong = btn.querySelector("strong");
      if (!strong) return;
      const code = strong.textContent.trim();
      if (!/^[1-5]\d{2}$/.test(code)) return;
      buttons.push(idx);
    });
    return buttons;
  });

  console.log(`Step 1: Expanding ${responseButtons.length} response buttons in batches of ${BATCH_SIZE}...`);

  // Click in batches to avoid overwhelming React
  for (let i = 0; i < responseButtons.length; i += BATCH_SIZE) {
    const batch = responseButtons.slice(i, i + BATCH_SIZE);
    await page.evaluate((indices) => {
      const allBtns = document.querySelectorAll("button");
      indices.forEach((idx) => {
        const btn = allBtns[idx];
        if (!btn) return;
        const strong = btn.querySelector("strong");
        if (!strong) return;
        const code = strong.textContent.trim();
        if (!/^[1-5]\d{2}$/.test(code)) return;
        if (btn.parentElement.children.length <= 1) {
          btn.click();
        }
      });
    }, batch);
    await DELAY(600);
  }
  await DELAY(1000);

  // ── Step 2 ──────────────────────────────────────────────────────────────
  // For all switchable tab lists, ensure the first tab is selected.
  console.log("Step 2: Selecting first tab in all switchable tab lists...");
  await page.evaluate(() => {
    document.querySelectorAll("ul.react-tabs__tab-list").forEach((tabList) => {
      const tabs = tabList.querySelectorAll('li[role="tab"]');
      if (tabs.length === 0) return;

      const firstTab = tabs[0];
      if (!firstTab.classList.contains("react-tabs__tab--selected")) {
        firstTab.click();
      }
    });
  });
  await DELAY(1500);

  // ── Step 3 ──────────────────────────────────────────────────────────────
  // Expand nested schema properties and collapsible sections.
  //
  // IMPORTANT: ReDoc JSON samples use `button.collapser` (aria-label
  // toggles between "expand"/"collapse") and child `div.hoverable.collapsed`
  // elements. Clicking the collapser expands the JSON node; clicking the
  // hoverable div re-collapses it. So we must:
  //   a) Click only `button.collapser[aria-label="expand"]` (not the divs)
  //   b) Remove the "collapsed" class from `div.hoverable` via DOM, not click
  // The broad `[class*="collapsed"]` selector caused an infinite toggle loop.
  console.log("Step 3: Expanding nested schema properties...");
  for (let iteration = 0; iteration < 5; iteration++) {
    const expandedCount = await page.evaluate(() => {
      let count = 0;

      // Expand [role="button"] elements with aria-expanded="false"
      document.querySelectorAll('[role="button"]').forEach((el) => {
        if (el.getAttribute("aria-expanded") === "false") {
          el.click();
          count++;
        }
      });

      // Expand <details> elements
      document.querySelectorAll("details:not([open])").forEach((el) => {
        el.setAttribute("open", "");
        count++;
      });

      // Expand <summary> elements inside closed <details>
      document.querySelectorAll("summary").forEach((el) => {
        if (el.parentElement && !el.parentElement.open) {
          el.click();
          count++;
        }
      });

      // Click JSON sample collapsers that are in "expand" state
      document
        .querySelectorAll('button.collapser[aria-label="expand"]')
        .forEach((el) => {
          el.click();
          count++;
        });

      return count;
    });

    await DELAY(800);
    if (expandedCount === 0) break;
    console.log(`  iteration ${iteration + 1}: expanded ${expandedCount} elements`);
  }

  // ── Step 4 ──────────────────────────────────────────────────────────────
  // Force-expand all remaining collapsed JSON sample nodes by removing the
  // "collapsed" class via DOM manipulation (NOT clicking, which would toggle).
  console.log("Step 4: Force-expanding collapsed JSON sample nodes...");
  const removedCount = await page.evaluate(() => {
    let count = 0;
    document.querySelectorAll(".hoverable.collapsed").forEach((el) => {
      el.classList.remove("collapsed");
      count++;
    });
    return count;
  });
  console.log(`  removed 'collapsed' class from ${removedCount} elements`);
  await DELAY(500);

  // Force visible CSS
  await page.addStyleTag({
    content: `
      * {
        visibility: visible !important;
        max-height: none !important;
        overflow: visible !important;
      }
      .collapsible {
        height: auto !important;
        display: block !important;
      }
      .collapsed {
        display: block !important;
      }
      .hoverable.collapsed {
        display: block !important;
      }
    `,
  });

  // Export PDF via CDP directly to avoid Puppeteer's timeout wrapper
  console.log("Generating PDF (this may take several minutes for large specs)...");
  const cdp = await page.createCDPSession();
  const { data } = await cdp.send("Page.printToPDF", {
    landscape: false,
    printBackground: true,
    paperWidth: 8.27,   // A4 width in inches
    paperHeight: 11.69, // A4 height in inches
    marginTop: 0.59,    // 15mm in inches
    marginBottom: 0.59,
    marginLeft: 0.47,   // 12mm in inches
    marginRight: 0.47,
  });
  fs.writeFileSync(pdfPath, Buffer.from(data, "base64"));

  await browser.close();
  console.log(`PDF exported successfully: ${pdfPath}`);
}

exportPDF();
