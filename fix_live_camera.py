"""
Patch script: replaces the page_live() function in 05_dashboard.py
with a robust version that handles camera timeouts and permission errors.
"""
import re

DASHBOARD = r"c:\Users\ACER\Desktop\INTERNSHIP PROJECT\FaceAttendance\05_dashboard.py"

with open(DASHBOARD, "r", encoding="utf-8") as f:
    src = f.read()

NEW_FUNC = r'''def page_live():
    body = """
<div style="max-width:900px">
<h2 style="margin-bottom:6px;font-size:20px">Live Web Attendance</h2>
<p style="color:var(--muted);font-size:14px;margin-bottom:24px">
  Start the camera to begin marking attendance automatically.
  Students recognized will be saved to the database.
</p>

<div class="card" style="padding:24px">
  <div class="reg-grid">
    <div>
      <div class="cam-box">
        <video id="video" autoplay playsinline muted></video>
        <canvas id="canvas" width="320" height="240" style="display:none"></canvas>
        <div class="cam-overlay" id="cam_status_text">Camera Off</div>
      </div>
      <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
        <button class="btn btn-primary" id="btn_start_cam" onclick="startCameraAndScan()">&#9654; Start Camera &amp; Scan</button>
        <button class="btn btn-outline" id="btn_stop" onclick="stopAll()" style="display:none">&#9632; Turn Off Camera</button>
      </div>
      <div id="cam_help" style="display:none;margin-top:12px;font-size:12px;color:var(--muted);line-height:1.7;background:rgba(255,255,255,0.03);border:1px solid var(--border);border-radius:8px;padding:12px">
        <strong style="color:var(--yellow)">&#9888; Camera Tips:</strong><br>
        &bull; Click the camera/lock icon in your browser address bar and allow camera<br>
        &bull; Make sure no other app or tab is using the camera<br>
        &bull; Try refreshing the page after granting permission<br>
        &bull; Use Chrome or Edge for best compatibility
      </div>
    </div>

    <div>
      <h3 style="font-size:15px;color:var(--accent);margin-bottom:12px">Last Recognized</h3>
      <div class="status-box info" id="scan_status" style="display:block;margin-bottom:16px">Waiting to start...</div>
      <div id="recognized_list" style="display:flex;flex-direction:column;gap:8px;max-height:200px;overflow-y:auto">
      </div>
    </div>
  </div>
</div>
</div>

<script>
var stream = null;
var scanning = false;
var lastRecognized = "";

function setStatus(msg, type) {
  var el = document.getElementById('scan_status');
  el.textContent = msg;
  el.className = 'status-box ' + type;
  el.style.display = 'block';
}

// Progressive fallback: tries strict -> relaxed -> bare video:true
function tryGetUserMedia() {
  var constraintsList = [
    { video: { width: { ideal: 320 }, height: { ideal: 240 }, facingMode: 'user' }, audio: false },
    { video: { facingMode: 'user' }, audio: false },
    { video: true, audio: false }
  ];
  function tryNext(idx) {
    if (idx >= constraintsList.length) {
      return Promise.reject(new Error('Could not access camera with any settings.'));
    }
    return navigator.mediaDevices.getUserMedia(constraintsList[idx]).catch(function(e) {
      if (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError') {
        return Promise.reject(e);
      }
      return tryNext(idx + 1);
    });
  }
  return tryNext(0);
}

// Wait for video element to have real frames (readyState >= 2 and videoWidth > 0)
function waitForVideoReady(video) {
  var timeoutMs = 8000;
  return new Promise(function(resolve, reject) {
    if (video.readyState >= 2 && video.videoWidth > 0) { resolve(); return; }
    var deadline = Date.now() + timeoutMs;
    var timer = setInterval(function() {
      if (video.readyState >= 2 && video.videoWidth > 0) {
        clearInterval(timer); resolve();
      } else if (Date.now() > deadline) {
        clearInterval(timer);
        reject(new Error('Camera timed out. Please close other apps using the camera and try again.'));
      }
    }, 100);
  });
}

function startCameraAndScan() {
  var btn = document.getElementById('btn_start_cam');
  btn.disabled = true;
  btn.textContent = 'Starting camera...';
  document.getElementById('cam_help').style.display = 'none';
  setStatus('Requesting camera access...', 'info');

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus('\u274C Camera API not available. Use Chrome or Edge over HTTPS.', 'error');
    btn.disabled = false;
    btn.textContent = '\u25BA Start Camera & Scan';
    document.getElementById('cam_help').style.display = 'block';
    return;
  }

  tryGetUserMedia().then(function(s) {
    stream = s;
    var video = document.getElementById('video');
    video.srcObject = stream;
    setStatus('Camera starting, please wait...', 'info');
    document.getElementById('cam_status_text').textContent = 'Loading...';
    return waitForVideoReady(video);
  }).then(function() {
    var btn2 = document.getElementById('btn_start_cam');
    btn2.style.display = 'none';
    document.getElementById('btn_stop').style.display = 'inline-block';
    document.getElementById('cam_status_text').textContent = 'Scanning...';
    startScanning();
  }).catch(function(e) {
    var btn2 = document.getElementById('btn_start_cam');
    btn2.disabled = false;
    btn2.textContent = '\u25BA Start Camera & Scan';
    document.getElementById('cam_status_text').textContent = 'Camera Off';
    document.getElementById('cam_help').style.display = 'block';

    var msg = 'Camera error: ' + e.message;
    if (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError') {
      msg = '\u274C Permission denied. Click the lock/camera icon in your address bar, allow camera, then refresh.';
    } else if (e.name === 'NotFoundError' || e.name === 'DevicesNotFoundError') {
      msg = '\u274C No camera found on this device.';
    } else if (e.name === 'NotReadableError' || e.name === 'TrackStartError') {
      msg = '\u274C Camera is in use by another app or tab. Close it and try again.';
    } else if (e.message && e.message.indexOf('timed out') !== -1) {
      msg = '\u274C ' + e.message;
    }
    setStatus(msg, 'error');
    if (stream) { stream.getTracks().forEach(function(t) { t.stop(); }); stream = null; }
  });
}

function captureLiveFrame() {
  var video = document.getElementById('video');
  var canvas = document.getElementById('canvas');
  canvas.width = 320;
  canvas.height = 240;
  var ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, 320, 240);
  var fullData = ctx.getImageData(0, 0, 320, 240).data;
  var totalSkin = 0;
  for (var i = 0; i < fullData.length; i += 4) {
    var r = fullData[i], g = fullData[i+1], b = fullData[i+2];
    if (r > 30 && g > 15 && b > 5 && r > b && (r - b) > 5 && r < 255) totalSkin++;
  }
  var faceFound = (totalSkin / (fullData.length / 4)) > 0.02;
  var dataUrl = canvas.toDataURL('image/jpeg', 0.85);
  return { dataUrl: dataUrl, faceFound: faceFound };
}

function sleep(ms) { return new Promise(function(r) { setTimeout(r, ms); }); }

async function startScanning() {
  scanning = true;
  setStatus('Look straight at the camera...', 'info');
  while (scanning) {
    var result = captureLiveFrame();
    if (!result.faceFound) {
      setStatus('Camera covered or too dark \u2014 face the camera', 'info');
      lastRecognized = "";
      await sleep(300);
      continue;
    }
    try {
      var res = await fetch('/api/recognize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: result.dataUrl })
      });
      var data = await res.json();
      if (data.ok && data.name && data.name !== "Unknown") {
        if (data.already_marked) {
          setStatus('\u2705 ' + data.name + ' \u2014 Already marked present today', 'success');
        } else if (data.name !== lastRecognized) {
          lastRecognized = data.name;
          setStatus('\u2705 ' + data.name + ' marked Present!', 'success');
          var list = document.getElementById('recognized_list');
          var item = document.createElement('div');
          var now = new Date();
          var timeStr = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
          item.style.cssText = 'padding:10px;background:var(--card);border:1px solid var(--green);border-radius:8px;display:flex;justify-content:space-between;align-items:center';
          item.innerHTML = '<strong>' + data.name + '</strong><span style="color:var(--muted);font-size:12px">' + timeStr + '</span><span class="pill present">Present</span>';
          list.prepend(item);
        }
      } else if (data.ok && data.pending) {
        setStatus('Verifying: ' + data.pending + ' (' + data.frames + '/' + (data.needed || 5) + ' frames, score: ' + data.conf + ')', 'info');
        lastRecognized = "";
      } else if (data.ok && data.name === "Unknown") {
        setStatus('Face detected but not recognized (Score: ' + Math.round(data.conf) + ' \u2014 need < 60 to match)', 'error');
        lastRecognized = "";
      } else if (data.ok && data.name === null) {
        setStatus('No face detected \u2014 look straight at the camera', 'info');
        lastRecognized = "";
      } else if (!data.ok) {
        setStatus('\u274C ' + data.error, 'error');
        scanning = false;
        break;
      }
    } catch(e) { /* retry silently */ }
    await sleep(150);
  }
}

function stopAll() {
  scanning = false;
  lastRecognized = "";
  if (stream) { stream.getTracks().forEach(function(t) { t.stop(); }); stream = null; }
  var btn = document.getElementById('btn_start_cam');
  btn.disabled = false;
  btn.textContent = '\u25BA Start Camera & Scan';
  btn.style.display = 'inline-block';
  document.getElementById('btn_stop').style.display = 'none';
  document.getElementById('cam_status_text').textContent = 'Camera Off';
  setStatus('Camera stopped. Press Start to scan again.', 'info');
}
</script>
"""
    return html_page("Live Attendance", body, "Live Attendance")
'''

# Find and replace the page_live function
pattern = re.compile(
    r'def page_live\(\):.*?return html_page\("Live Attendance", body, "Live Attendance"\)',
    re.DOTALL
)

match = pattern.search(src)
if not match:
    print("ERROR: Could not find page_live() function!")
    exit(1)

new_src = src[:match.start()] + NEW_FUNC + src[match.end():]

with open(DASHBOARD, "w", encoding="utf-8") as f:
    f.write(new_src)

print("SUCCESS: page_live() replaced successfully.")
print(f"Original function was {match.end()-match.start()} chars, new is {len(NEW_FUNC)} chars.")
