import { useEffect, useRef, useState } from 'react';

export const DISCLAIMER = 'Model score for one Stable Diffusion v1.5 dataset; not proof of authenticity; may be wrong on other generators, screenshots or compressed images.';
export const FEATURE_ONLY_NOTICE = 'Research model not connected. AI vs Real prediction is unavailable. The visualizations below show model-independent frequency analysis.';
const FORMATS = ['image/jpeg', 'image/png', 'image/webp'];

function Icon({ name = 'arrow', size = 20, ...props }) {
  const paths = { arrow: 'M4 12h16m-6-6 6 6-6 6', upload: 'M12 16V4m-5 5 5-5 5 5M4 16v4h16v-4',
    image: 'M3 4h18v16H3zM3 16l5-5 5 5 3-3 5 5M15 8h.01',
    lock: 'M6 10h12v11H6zM8 10V7a4 4 0 0 1 8 0v3', close: 'm6 6 12 12M6 18 18 6',
    check: 'm5 12 4 4L19 6', info: 'M12 11v6m0-10v.01', refresh: 'M20 10a8 8 0 1 0-1 8M20 4v6h-6' };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}><path d={paths[name]} /></svg>;
}

function FrequencyMark({ large = false }) {
  return <div className={large ? 'frequency-art' : 'brand-mark'} aria-hidden="true">{[22, 40, 60, 34, 50, 26, 38].map((height, index) => <span key={index} style={{ '--bar': `${height}%` }} />)}</div>;
}

export function RapsdPlot({ data }) {
  const values = data.log_power;
  const min = Math.min(...values), max = Math.max(...values), range = max - min || 1;
  const points = values.map((value, i) => `${42 + i / Math.max(1, values.length - 1) * 458},${137 - (value - min) / range * 112}`).join(' ');
  return <figure className="rapsd-figure"><svg viewBox="0 0 526 174" role="img" aria-label="Radially averaged power spectrum. The horizontal axis shows spatial frequency; the vertical axis shows log power.">
    {[25, 81, 137].map((y, i) => <g key={y}><line x1="42" x2="500" y1={y} y2={y} stroke="#e5e9f1" /><text x="32" y={y + 4} textAnchor="end">{(max - i * (max - min) / 2).toFixed(1)}</text></g>)}
    <polyline points={points} fill="none" stroke="#3559db" strokeWidth="2.5" strokeLinejoin="round" />
    <text x="42" y="158">{data.radius[0].toFixed(0)}</text><text x="500" y="158" textAnchor="end">{data.radius.at(-1).toFixed(0)}</text>
  </svg><figcaption>Spatial frequency (cycles / image) · log power</figcaption></figure>;
}

function Result({ result, busy }) {
  if (!result) return <div className="empty-result" aria-busy={busy}>
    <FrequencyMark large />
    <span className="eyebrow">{busy ? 'READING THE FREQUENCIES' : 'BEYOND WHAT MEETS THE EYE'}</span>
    <h3>{busy ? 'Looking a little closer…' : 'Every image has a frequency signature.'}</h3>
    <p>{busy ? 'Preparing your image and measuring its spectral patterns.' : 'Add an image to explore its patterns and see how the research model classifies it.'}</p>
    <div className="empty-steps"><span><b>01</b> Prepare image</span><span><b>02</b> Inspect spectrum</span><span><b>03</b> Estimate class</span></div>
  </div>;
  if (result.mode === 'feature_analysis') return <div className="result-content">
    <div className="verdict"><span className="eyebrow">FEATURE ANALYSIS MODE</span><h3>Feature analysis only</h3><p>No classification model was used for this analysis.</p></div>
    <div className="notice feature-notice" role="status"><Icon name="info" /><p>{FEATURE_ONLY_NOTICE}</p></div>
    <div className="feature-stats" aria-label="Extracted image features">
      <div><span>Processed image</span><strong>{result.processed_size} × {result.processed_size}</strong></div>
      <div><span>Spectral entropy</span><strong>{Number.isFinite(result.feature_analysis?.spectral_entropy) ? result.feature_analysis.spectral_entropy.toFixed(4) : '—'}</strong></div>
      <div><span>RAPSD rings</span><strong>{result.feature_analysis?.rapsd_ring_count ?? '—'}</strong></div>
    </div>
    <div className="spectrum-grid"><figure className="spectrum-figure"><img src={result.spectrum_png} alt="Centered log-power spectrum of the uploaded image. Brighter regions indicate greater power." /><figcaption>Log-power spectrum</figcaption></figure><div className="spectrum-copy"><span className="eyebrow">FREQUENCY VIEW</span><h4>A different view of the same image.</h4><p>The center represents broad shapes. Patterns farther out represent finer detail.</p></div></div>
    <div className="plot-title"><h4>Radial power profile</h4><span>RAPSD</span></div><RapsdPlot data={result.rapsd} />
  </div>;
  return <div className="result-content">
    <div className="verdict"><span className="eyebrow">MODEL CLASSIFICATION</span><h3>{result.label}</h3><p>The selected research model’s prediction for this image.</p></div>
    <div className="score-heading"><span>AI-generated probability</span><strong>{(result.ai_probability * 100).toFixed(1)}<small>%</small></strong></div>
    <div className="probability-track" role="meter" aria-label="AI-generated probability" aria-valuemin="0" aria-valuemax="100" aria-valuenow={result.ai_probability * 100}><span style={{ width: `${result.ai_probability * 100}%` }} /></div>
    <div className="score-endpoints"><span>Real</span><span>AI-generated</span></div>
    <p className="score-note">A model probability is not a measure of authenticity.</p>
    <div className="spectrum-grid"><figure className="spectrum-figure"><img src={result.spectrum_png} alt="Centered log-power spectrum of the uploaded image. Brighter regions indicate greater power." /><figcaption>Log-power spectrum</figcaption></figure><div className="spectrum-copy"><span className="eyebrow">FREQUENCY VIEW</span><h4>A different view of the same image.</h4><p>The center represents broad shapes. Patterns farther out represent finer detail.</p></div></div>
    <div className="plot-title"><h4>Radial power profile</h4><span>RAPSD</span></div><RapsdPlot data={result.rapsd} />
    {result.warnings.length > 0 && <div className="notice warning" role="status"><Icon name="info" /><div><strong>Library versions differ</strong><p>Model compatibility may affect this result.</p><details><summary>View version details</summary><ul>{result.warnings.map(item => <li key={item}>{item}</li>)}</ul></details></div></div>}
  </div>;
}

export default function App() {
  const [health, setHealth] = useState(null), [offline, setOffline] = useState(false);
  const [file, setFile] = useState(null), [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null), [error, setError] = useState('');
  const [busy, setBusy] = useState(false), [dragging, setDragging] = useState(false), [about, setAbout] = useState(false);
  const input = useRef(null), request = useRef(null), sequence = useRef(0), resultHeading = useRef(null);

  async function checkHealth(signal) {
    try {
      const response = await fetch('/api/health', { signal });
      if (!response.ok) throw new Error();
      const value = await response.json();
      if (!signal?.aborted) { setHealth(value); setOffline(false); }
    } catch (err) { if (err.name !== 'AbortError') setOffline(true); }
  }
  useEffect(() => { const controller = new AbortController(); checkHealth(controller.signal); return () => { controller.abort(); request.current?.abort(); }; }, []);
  useEffect(() => { if (!file) { setPreview(null); return; } const url = URL.createObjectURL(file); setPreview({ file, url }); return () => URL.revokeObjectURL(url); }, [file]);

  function choose(files) {
    sequence.current += 1;
    request.current?.abort(); setBusy(false); setResult(null); setError('');
    if (!files?.length) return;
    setFile(null);
    if (files.length !== 1) { setError('Choose one image at a time.'); return; }
    const selected = files[0];
    if (!FORMATS.includes(selected.type)) { setError('Choose a JPEG, PNG, or WebP image.'); return; }
    if (selected.size > (health?.max_upload_bytes ?? 10485760)) { setError('Choose an image smaller than 10 MB.'); return; }
    setFile(selected);
    if (input.current) input.current.value = '';
  }
  function clear() { sequence.current += 1; request.current?.abort(); setBusy(false); setFile(null); setResult(null); setError(''); }
  async function analyze() {
    if (!file || busy) return;
    const id = ++sequence.current;
    request.current?.abort(); const controller = new AbortController(); request.current = controller;
    setBusy(true); setError(''); setResult(null);
    try {
      const response = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': file.type }, body: file, signal: controller.signal });
      let data;
      try { data = await response.json(); }
      catch { throw new Error('The local service could not complete the analysis. Please try again.'); }
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'This image could not be analyzed. Try another image.');
      if (sequence.current === id) { setResult(data); resultHeading.current?.focus(); }
    } catch (err) { if (err.name !== 'AbortError' && sequence.current === id) setError(err.message === 'Failed to fetch' ? 'The local service disconnected. Check the connection and try again.' : err.message); }
    finally { if (sequence.current === id) setBusy(false); }
  }
  const featureAnalysisReady = health?.feature_analysis_available && !offline;
  return <div className="app-shell">
    <a className="skip-link" href="#workspace">Skip to image analysis</a>
    <header className="topbar"><a className="brand" href="#workspace" aria-label="SpectraCheck home"><FrequencyMark /><span>Spectra<span className="brand-light">Check</span><small>IMAGE FORENSICS LAB</small></span></a>
      <nav aria-label="Main navigation"><button className={!about ? 'nav-link active' : 'nav-link'} onClick={() => setAbout(false)} aria-current={!about ? 'page' : undefined}>Image analysis</button><button className={about ? 'nav-link active' : 'nav-link'} onClick={() => setAbout(true)} aria-current={about ? 'page' : undefined}>About the study</button></nav>
      <span className="session"><span className={offline ? 'dot amber' : 'dot'} />Local session</span>
    </header>
    <main id="workspace">
      <section className="page-heading"><span className="eyebrow blue">FREQUENCY-DOMAIN IMAGE ANALYSIS</span><h1>A closer look.<br className="mobile-break" /> A clearer perspective.</h1><p>Explore the patterns behind the pixels, with a research model<br className="desktop-break" /> trained to distinguish real and AI-generated images.</p></section>
      {about ? <section className="about-panel panel"><span className="eyebrow blue">THE STUDY</span><h2>Evidence in the frequency domain.</h2><p>SpectraCheck studies handcrafted 2D FFT features from real photographs and Stable Diffusion 1.5 images. Support Vector Machine, Random Forest, and XGBoost classifiers are compared with a pixel baseline.</p><p>The demo uses the frequency model selected on validation accuracy. Frequency patterns support a model prediction; they cannot establish an image’s authenticity.</p><div className="about-columns"><div><h3>Research scope</h3><p>One generator and one dataset. Other generators, phone photos, screenshots, and compressed images may behave differently.</p></div><div><h3>Privacy by design</h3><p>Images are sent only to the service on this computer, processed in memory, and never saved by the app.</p></div></div><button className="primary" onClick={() => setAbout(false)}>Analyze an image <Icon /></button></section> : <>
      <div className="workspace-grid">
        <section className="upload-panel panel" aria-labelledby="upload-title"><div className="section-heading"><span className="step-number">01</span><h2 id="upload-title">Your image</h2><span className="section-tag">UPLOAD</span></div>
          <input className="visually-hidden" type="file" id="image-upload" aria-label="Choose an image" aria-hidden="true" accept="image/jpeg,image/png,image/webp" ref={input} onChange={event => choose(event.target.files)} tabIndex={-1} />
          <div className={`dropzone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`} onDragOver={event => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); choose(event.dataTransfer.files); }}>
            {file ? <div className="preview-wrap">{preview?.file === file && <img className="image-preview" src={preview.url} alt={`Preview of ${file.name}`} onError={() => { setError('This file could not be previewed. Choose a valid image.'); setFile(null); }} />}</div> : <><span className="upload-icon"><Icon name="upload" size={26} /></span><h3>Bring an image.<br />We’ll look deeper.</h3><p>Drag and drop your image here</p><span className="or-divider">or</span></>}
            <button className="secondary" onClick={() => input.current?.click()}>{file ? 'Change image' : 'Choose an image'} <Icon name="image" size={17} /></button>
            {!file && <small>JPEG, PNG, WebP · up to 10 MB<br />At least {health?.min_image_side ?? 256} × {health?.min_image_side ?? 256} pixels</small>}
          </div>
          {file && <div className="file-row"><span className="file-symbol"><Icon name="image" /></span><div><strong title={file.name}>{file.name}</strong><span>{(file.size / 1048576).toFixed(2)} MB · Ready to inspect</span></div><button className="icon-button" onClick={clear} aria-label="Remove image"><Icon name="close" size={18} /></button></div>}
          {error && <div className="notice error" role="alert"><Icon name="info" /><p>{error}</p></div>}
          <button className="primary analyze-button" disabled={!file || !featureAnalysisReady || busy} onClick={analyze} aria-describedby="service-note">{busy ? 'Analyzing image…' : 'Analyze image'}<Icon /></button>
          <p className="privacy-note"><Icon name="lock" size={13} />Your image stays on this computer.</p>
          {!featureAnalysisReady && <div className="service-note" id="service-note" role="status"><span className="dot amber" /><div><strong>{offline ? 'Local service is offline' : health ? 'Feature analysis not available' : 'Connecting to local service…'}</strong><p>{offline ? 'Start the local service, then reconnect.' : health ? (health.feature_analysis_message ?? 'The local feature-analysis configuration is unavailable.') : 'Image preview is available while the local service connects.'}</p><button className="text-button" onClick={() => checkHealth()}>Check connection <Icon name="refresh" size={13} /></button></div></div>}
        </section>
        <section className="analysis-panel panel" aria-labelledby="result-title"><div className="section-heading"><span className="step-number">02</span><h2 id="result-title" tabIndex={-1} ref={resultHeading}>The analysis</h2><span className={`result-status ${result ? 'complete' : ''}`} role="status">{busy ? 'Analyzing' : result ? 'Complete' : file ? 'Ready to analyze' : 'Awaiting image'}</span></div><Result result={result} busy={busy} /></section>
      </div>
      <aside className="scope-note"><span className="scope-icon"><Icon name="info" size={17} /></span><p><strong>A research perspective, not a verdict.</strong> {health?.disclaimer ?? DISCLAIMER}</p><span className="scope-tag">SD 1.5 DATASET</span></aside>
      <section className="method-strip" aria-label="How it works"><div><span>01 / PREPARE</span><h3>One consistent view.</h3><p>The same grayscale preparation used during training.</p></div><div><span>02 / TRANSFORM</span><h3>Patterns beyond pixels.</h3><p>A frequency spectrum captures fine and broad image detail.</p></div><div><span>03 / INTERPRET</span><h3>A measured prediction.</h3><p>A validation-selected model returns its class probability.</p></div></section>
      </>}
    </main>
    <footer><span>© {new Date().getFullYear()} SpectraCheck</span><span>An MCA research project · MET Institute of Computer Science</span><span>LOCAL RESEARCH PROTOTYPE</span></footer>
  </div>;
}
