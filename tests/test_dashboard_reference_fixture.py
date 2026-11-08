"""Real Chromium checks of the restored dashboard assets and retained controls."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from chronovisor.ops import dashboard


def _chrome() -> str:
    candidate = next(
        (
            value
            for value in (
                shutil.which("google-chrome"),
                shutil.which("chromium"),
                shutil.which("chromium-browser"),
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            )
            if value and Path(value).is_file()
        ),
        None,
    )
    assert candidate is not None, "headless Chrome is required for Dashboard DOM tests"
    return candidate


_BROWSER_CHECK = r"""
const [endpoint, url, screenshot, width, expression, fullPage] = process.argv.slice(2);
const fs = await import('node:fs/promises');
const targets = await (await fetch(endpoint + '/json/list')).json();
const target = targets.find(target => target.type === 'page');
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
let sequence = 0;
const waiting = new Map();
const exceptions = [];
const failures = [];
const requests = [];
ws.onmessage = ({data}) => {
  const message = JSON.parse(data);
  if (message.id) {
    const pair = waiting.get(message.id);
    waiting.delete(message.id);
    if (message.error) pair.reject(new Error(JSON.stringify(message.error)));
    else pair.resolve(message.result);
  } else if (message.method === 'Runtime.exceptionThrown') {
    exceptions.push(message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text);
  } else if (message.method === 'Network.responseReceived') {
    const response = message.params.response;
    if (response.url.startsWith(new URL(url).origin)) {
      requests.push({url: new URL(response.url).pathname, status: response.status});
      if (response.status >= 400 && !response.url.endsWith('/favicon.ico')) failures.push(response.url + ': ' + response.status);
    }
  } else if (message.method === 'Log.entryAdded' && message.params.entry.level === 'error') {
    const entry = message.params.entry;
    if (!entry.url?.endsWith('/favicon.ico')) exceptions.push(entry.text);
  }
};
const call = (method, params={}) => new Promise((resolve,reject) => {
  const id = ++sequence;
  waiting.set(id, {resolve,reject});
  ws.send(JSON.stringify({id, method, params}));
});
try {
  await call('Runtime.enable');
  await call('Network.enable');
  await call('Log.enable');
  await call('Page.enable');
  await call('Emulation.setDeviceMetricsOverride', {width:Number(width), height:2400, deviceScaleFactor:1, mobile:false});
  await call('Emulation.setEmulatedMedia', {features:[{name:'prefers-reduced-motion',value:'reduce'}]});
  await call('Page.navigate', {url});
  const result = await call('Runtime.evaluate', {
    expression:`(async () => {
      const deadline = Date.now() + 20000;
      while (!(document.body?.dataset.snapshotState === 'full' || (window.chronovisorCortexMetrics?.().nodeCount >= 0 && document.getElementById('stN')?.textContent !== '—'))) {
        if (document.querySelector('.bootError')) throw new Error(document.querySelector('.bootError').textContent);
        if (Date.now() > deadline) throw new Error('Dashboard did not become ready');
        await new Promise(resolve => setTimeout(resolve, 50));
      }
      await document.fonts.ready;
      ${expression}
    })()`, awaitPromise:true, returnByValue:true,
  });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
  const layout = await call('Page.getLayoutMetrics');
  const clip = fullPage === 'true' ? {...layout.cssContentSize, scale:1} : undefined;
  const image = await call('Page.captureScreenshot', {format:'png', captureBeyondViewport:!!clip, clip});
  await fs.writeFile(screenshot, Buffer.from(image.data, 'base64'));
  console.log(JSON.stringify({audit:result.result.value, exceptions, failures, requests}));
} finally { ws.close(); }
"""


def check_browser(
    url: str,
    screenshot: Path,
    tmp_path: Path,
    *,
    width: int = 1600,
    expression: str = "return {};",
    full_page: bool = False,
) -> dict:
    """Use the existing Chromium fixture against synthetic or production data."""
    profile = tmp_path / "chrome-profile"
    profile.mkdir(parents=True)
    script = tmp_path / "browser-check.mjs"
    script.write_text(_BROWSER_CHECK)
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            _chrome(),
            "--headless=new",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-background-networking",
            "--disable-component-update",
            "--no-first-run",
            "--no-default-browser-check",
            "--remote-debugging-port=0",
            f"--user-data-dir={profile}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        endpoint_file = profile / "DevToolsActivePort"
        deadline = time.monotonic() + 10
        while not endpoint_file.exists():
            assert time.monotonic() < deadline and process.poll() is None, (
                "Chrome failed to start"
            )
            time.sleep(0.05)
        port = endpoint_file.read_text().splitlines()[0]
        completed = subprocess.run(
            [
                "node",
                str(script),
                f"http://127.0.0.1:{port}",
                url,
                str(screenshot),
                str(width),
                expression,
                str(full_page).lower(),
            ],
            text=True,
            capture_output=True,
            timeout=40,
        )
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        assert not result["exceptions"], result
        assert not result["failures"], result
        assert screenshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        return result
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@contextmanager
def _preview(snapshot, graph=None):
    class Handler(dashboard.DashboardHandler):
        def do_GET(self):
            if self.path == "/api/snapshot":
                if self._browser_boundary_allows():
                    dashboard._json_response(self, snapshot)
                return
            if self.path == "/api/cortex/graph":
                if self._browser_boundary_allows():
                    dashboard._json_response(self, graph)
                return
            if self.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            super().do_GET()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


OBSERVATORY_AUDIT = r"""
const required = ['Save History', 'Save Load', 'Knowledge Mix', 'Memory Health', 'Model Fleet', 'Recent Events'];
const headings = [...document.querySelectorAll('h2')].map(node => node.textContent);
if (JSON.stringify(headings) !== JSON.stringify(required)) throw new Error('Unexpected panels: '+headings);
for (const mode of ['weekly','cumulative','daily']) document.querySelector(`[data-save-mode="${mode}"]`).click();
for (const mode of ['pages','size']) document.querySelector(`[data-knowledge-mode="${mode}"]`).click();
document.querySelector('.save-cell:not(.blank)')?.click();
await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
const overflowing = [...document.querySelectorAll('.metric-card, .model-row, .health-group, .event')].filter(node => node.scrollWidth > node.clientWidth+1).map(node => node.className);
if (document.documentElement.scrollWidth > window.innerWidth+1 || overflowing.length) throw new Error('Overflow: '+overflowing);
const metrics = [...document.querySelectorAll('.metric-card strong')].map(node => node.textContent);
if (metrics.length !== 5) throw new Error('Expected five metric cards');
return {headings, metrics, width:innerWidth, overflowing, modelCount:document.querySelectorAll('.model-row').length, disabled:document.querySelectorAll('.model-row.disabled').length, text:document.body.innerText};
"""


@pytest.mark.parametrize("width", [1600, 760])
@pytest.mark.parametrize(
    "state", ["ready", "queued", "running", "retry_wait", "failed", "empty", "disabled"]
)
def test_observatory_states_and_controls(tmp_path, monkeypatch, width, state):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", root)
    if state == "disabled":
        (root / "config.toml").write_text("[search.embedding]\nenabled = false\n")
    if state != "empty":
        (root / "raw").mkdir()
        day = datetime.now(dashboard.DASHBOARD_TIMEZONE).strftime("%Y%m%d")
        (root / "raw" / f"codex-{day}.md").write_text("raw capture")
        (root / "pages").mkdir()
        (root / "pages" / "frozen.md").write_text(
            "---\ntitle: Frozen\nstatus: stable\ntype: knowledge\n---\nPage"
        )
        runtime = root / "runtime"
        (runtime / "background-jobs").mkdir(parents=True)
        (runtime / "background-jobs/state.json").write_text(
            json.dumps(
                {
                    "jobs": {
                        "capture": {
                            "module": "chronovisor.hosts.codex_record",
                            "status": state
                            if state in {"queued", "running", "retry_wait", "failed"}
                            else "completed",
                            "updated_at": "2026-11-08T00:00:00+00:00",
                            "exit_code": 1 if state == "failed" else 0,
                        }
                    }
                }
            )
        )
        for service, model in (
            ("semantic", "nvidia/Nemotron-3-Embed-1B-BF16"),
            ("reranker", "StartLux-10B-search-rerank"),
        ):
            if state == "disabled" and service == "semantic":
                continue
            (runtime / f"{service}-service-status.json").write_text(
                json.dumps(
                    {
                        "model": model,
                        "status": "ok",
                        "ready": state != "failed",
                        "pid": os.getpid(),
                        "observed_at_epoch": time.time(),
                    }
                )
            )
    snapshot = dashboard.build_snapshot()
    visual = Path(os.environ.get("CHRONOVISOR_DASHBOARD_VISUAL_DIR", tmp_path))
    with _preview(snapshot) as url:
        result = check_browser(
            url,
            visual / f"dashboard-observatory-{state}-{width}.png",
            tmp_path,
            width=width,
            expression=OBSERVATORY_AUDIT,
            full_page=True,
        )
    assert result["audit"]["modelCount"] == 2
    assert result["audit"]["metrics"][0] == ("0" if state == "empty" else "1")
    assert result["audit"]["metrics"][1] == (
        "1" if state in {"queued", "retry_wait"} else "0"
    )
    if state == "running":
        assert "RUNNING" in result["audit"]["text"]
    if state == "disabled":
        assert result["audit"]["metrics"][-1] == "1 / 1"
        assert result["audit"]["disabled"] == 1
        assert "semantic: disabled" in result["audit"]["text"]


@pytest.mark.parametrize("mode", ["organic", "sphere", "cluster", "empty"])
def test_cortex_frozen_graph_controls_and_capture_preview(tmp_path, mode):
    nodes = (
        []
        if mode == "empty"
        else [
            {
                "id": f"page-{i}",
                "pkg": "memory",
                "l": 20 + i,
                "b": 100,
                "fi": 1,
                "fo": 1,
                "title": '<img src=x onerror="throw Error(1)">'
                if i == 0
                else f"Page {i}",
                "tags": ["frozen"],
            }
            for i in range(12)
        ]
    )
    graph = {
        "nodes": nodes,
        "links": [[i, (i + 1) % len(nodes), 0] for i in range(len(nodes))],
        "meta": {"static": len(nodes), "deferred": 0, "totalLines": 300},
        "categories": [{"id": "memory", "count": len(nodes)}] if nodes else [],
    }
    visual = Path(os.environ.get("CHRONOVISOR_DASHBOARD_VISUAL_DIR", tmp_path))
    expression = (
        """
const mode = """
        + json.dumps("organic" if mode == "empty" else mode)
        + r""";
document.getElementById({organic:'mOrganic',sphere:'mSphere',cluster:'mCluster'}[mode]).click();
if (Number(document.getElementById('stN').textContent)) {
 const search = document.getElementById('search');
 search.value = 'page-0'; search.dispatchEvent(new Event('input'));
 search.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter'}));
 if (!document.querySelector('#selCard .title').textContent.includes('<img')) throw new Error('Title escaping or selection failed');
 document.getElementById('closeSel').click();
 search.value = ''; search.dispatchEvent(new Event('input'));
 document.querySelector('[data-s="save"]').click();
}
await new Promise(resolve => setTimeout(resolve, 1000));
const metrics = window.chronovisorCortexMetrics();
if (metrics.mode !== mode) throw new Error('Mode switch failed');
if (document.querySelector('.bootError')) throw new Error('Graph boot failed');
if (document.documentElement.scrollWidth > innerWidth+1) throw new Error('Cortex overflow');
return {mode:metrics.mode, nodeCount:metrics.nodeCount, linkCount:metrics.linkCount, renderer:metrics.renderer, captures:metrics.visual.captureEvents};
"""
    )
    with _preview({}, graph) as url:
        result = check_browser(
            url + "/cortex",
            visual / f"dashboard-cortex-{mode}.png",
            tmp_path,
            expression=expression,
        )
    assert result["audit"]["nodeCount"] == len(nodes)
    assert result["audit"]["captures"] == (0 if mode == "empty" else 1)
