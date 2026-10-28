import React, { useState, useRef } from 'react';
import {
  Upload, AlertTriangle, AlertOctagon, CheckCircle2, RefreshCw,
  Copy, Check, ArrowRight, Eye, ShieldCheck, ShieldAlert, Sparkles,
  FileText, Activity, Stethoscope, ChevronDown, ChevronUp, Image as ImageIcon
} from 'lucide-react';
import './App.css';

export default function App() {
  // Upload and verification state
  const [frontalImage, setFrontalImage] = useState(null);
  const [lateralImage, setLateralImage] = useState(null);

  const [frontalStatus, setFrontalStatus] = useState('idle'); // 'idle' | 'validating' | 'valid' | 'invalid_not_xray' | 'invalid_not_chest'
  const [lateralStatus, setLateralStatus] = useState('idle');

  const [frontalMessage, setFrontalMessage] = useState('');
  const [lateralMessage, setLateralMessage] = useState('');

  const [frontalAnatomy, setFrontalAnatomy] = useState('');
  const [lateralAnatomy, setLateralAnatomy] = useState('');

  const [frontalView, setFrontalView] = useState('');
  const [lateralView, setLateralView] = useState('');

  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStep, setProcessingStep] = useState(1);
  const [resultData, setResultData] = useState(null);
  const [presetType, setPresetType] = useState(null);
  const [copied, setCopied] = useState(false);
  const [showAll14, setShowAll14] = useState(false);

  const frontalInputRef = useRef(null);
  const lateralInputRef = useRef(null);

  // Validate an image source (dataUrl or server path)
  const validateImage = async (imageSource, viewSlot = 'frontal') => {
    if (!imageSource) return;

    if (viewSlot === 'frontal') {
      setFrontalStatus('validating');
      setFrontalMessage('Verifying image modality and anatomy...');
    } else {
      setLateralStatus('validating');
      setLateralMessage('Verifying image modality and anatomy...');
    }

    try {
      const res = await fetch('/api/validate_image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_data: imageSource, view_hint: viewSlot }),
      });
      const data = await res.json();

      if (viewSlot === 'frontal') {
        if (!data.is_xray) {
          setFrontalStatus('invalid_not_xray');
          setFrontalMessage('Not an X-ray image. Please upload a valid medical X-ray.');
        } else if (!data.is_chest_xray) {
          setFrontalStatus('invalid_not_chest');
          setFrontalAnatomy(data.detected_anatomy || 'Non-Chest');
          setFrontalMessage(`Not a chest X-ray. Detected: ${data.detected_anatomy || 'Other body region'}. Please upload a chest radiograph.`);
        } else if (data.status === 'REJECTED_WRONG_VIEW' || data.detected_view === 'Lateral') {
          setFrontalStatus('invalid_wrong_view');
          setFrontalAnatomy('Chest');
          setFrontalView('Lateral');
          setFrontalMessage('Wrong Projection: This is a Lateral view. Slot 1 requires a Frontal (PA/AP) chest X-ray.');
        } else {
          setFrontalStatus('valid');
          setFrontalAnatomy('Chest');
          setFrontalView('Frontal');
          setFrontalMessage('Valid Frontal Chest Radiograph');
        }
      } else {
        if (!data.is_xray) {
          setLateralStatus('invalid_not_xray');
          setLateralMessage('Not an X-ray image. Please upload a valid medical X-ray.');
        } else if (!data.is_chest_xray) {
          setLateralStatus('invalid_not_chest');
          setLateralAnatomy(data.detected_anatomy || 'Non-Chest');
          setLateralMessage(`Not a chest X-ray. Detected: ${data.detected_anatomy || 'Other body region'}. Please upload a chest radiograph.`);
        } else if (data.status === 'REJECTED_WRONG_VIEW' || data.detected_view === 'Frontal') {
          setLateralStatus('invalid_wrong_view');
          setLateralAnatomy('Chest');
          setLateralView('Frontal');
          setLateralMessage('Wrong Projection: This is a Frontal view. Slot 2 requires a Lateral chest X-ray.');
        } else {
          setLateralStatus('valid');
          setLateralAnatomy('Chest');
          setLateralView('Lateral');
          setLateralMessage('Valid Lateral Chest Radiograph');
        }
      }
    } catch (err) {
      console.error('Validation error:', err);
      if (viewSlot === 'frontal') {
        setFrontalStatus('invalid_not_xray');
        setFrontalMessage('Could not verify image file. Please upload a valid image.');
      } else {
        setLateralStatus('invalid_not_xray');
        setLateralMessage('Could not verify image file. Please upload a valid image.');
      }
    }

  };

  // Handle local file selection
  const handleFileChange = (e, viewSlot) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const dataUrl = event.target.result;
      if (viewSlot === 'frontal') {
        setFrontalImage(dataUrl);
        validateImage(dataUrl, 'frontal');
      } else {
        setLateralImage(dataUrl);
        validateImage(dataUrl, 'lateral');
      }
    };
    reader.readAsDataURL(file);
    // Reset file input so re-uploading same file still fires
    e.target.value = '';
  };

  // Drag and drop handlers
  const handleDrop = (e, viewSlot) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const dataUrl = event.target.result;
      if (viewSlot === 'frontal') {
        setFrontalImage(dataUrl);
        validateImage(dataUrl, 'frontal');
      } else {
        setLateralImage(dataUrl);
        validateImage(dataUrl, 'lateral');
      }
    };
    reader.readAsDataURL(file);
  };

  // Swap frontal and lateral images if user loaded them inverted
  const handleSwap = () => {
    const tempF = frontalImage;
    const tempL = lateralImage;
    setFrontalImage(tempL);
    setLateralImage(tempF);
    if (tempL) validateImage(tempL, 'frontal');
    if (tempF) validateImage(tempF, 'lateral');
  };

  const isSwapped = frontalStatus === 'invalid_wrong_view' && lateralStatus === 'invalid_wrong_view';


  // Run full analysis & cross-checking
  const handleRunAnalysis = async () => {
    if (frontalStatus !== 'valid' || lateralStatus !== 'valid') return;

    setIsProcessing(true);
    setProcessingStep(1);

    // Visual step sequence
    const t1 = setTimeout(() => setProcessingStep(2), 650);
    const t2 = setTimeout(() => setProcessingStep(3), 1300);

    try {
      const res = await fetch('/api/process_radiographs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frontal_data: frontalImage,
          lateral_data: lateralImage,
          preset_type: presetType,
        }),
      });
      const data = await res.json();
      clearTimeout(t1);
      clearTimeout(t2);

      if (data.success) {
        setResultData(data);
      } else {
        alert(data.message || 'Error processing radiographs.');
      }
    } catch (err) {
      console.error('Processing failed:', err);
      alert('Failed to connect to diagnostic server.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Reset to upload screen
  const handleReset = () => {
    setFrontalImage(null);
    setLateralImage(null);
    setFrontalStatus('idle');
    setLateralStatus('idle');
    setFrontalMessage('');
    setLateralMessage('');
    setFrontalAnatomy('');
    setLateralAnatomy('');
    setFrontalView('');
    setLateralView('');
    setResultData(null);
    setPresetType(null);
    setCopied(false);
  };

  // Copy final report
  const copyReport = () => {
    if (!resultData?.final_verified_report) return;
    navigator.clipboard.writeText(resultData.final_verified_report);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isReadyToAnalyze = frontalStatus === 'valid' && lateralStatus === 'valid' && !isProcessing;

  return (
    <div className="min-app-container">
      {/* Header */}
      <header className="min-header">
        <div className="min-header-content">
          <div className="min-brand">
            <div className="min-brand-icon">
              <Stethoscope size={24} />
            </div>
            <div>
              <h1 className="min-title">Chest Radiograph AI Diagnosis</h1>
              <p className="min-subtitle">Dual-View Verification, Classification & Clinical Cross-Checking</p>
            </div>
          </div>
          {resultData && (
            <button className="min-btn-outline" onClick={handleReset}>
              <RefreshCw size={15} /> Upload New Images
            </button>
          )}
        </div>
      </header>

      {/* Main Body */}
      <main className="min-main">
        {!resultData ? (
          /* ======================================================= */
          /* STAGE 1: UPLOAD & VERIFICATION SCREEN                   */
          /* ======================================================= */
          <div className="min-upload-section">
            <div className="min-section-heading">
              <h2>Step 1: Upload Frontal and Lateral Chest X-Rays</h2>
              <p>Both radiographs must be verified medical chest X-rays to proceed to classification and reporting.</p>
            </div>

            <div className="min-upload-grid">
              {/* Box 1: Frontal Radiograph */}
              <div
                className={`min-upload-card ${
                  frontalStatus === 'valid'
                    ? 'card-valid'
                    : frontalStatus === 'invalid_not_xray' || frontalStatus === 'invalid_not_chest' || frontalStatus === 'invalid_wrong_view'
                    ? 'card-error'
                    : frontalStatus === 'validating'
                    ? 'card-validating'
                    : ''
                }`}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => handleDrop(e, 'frontal')}
              >
                <div className="min-card-top">
                  <div className="min-card-title">
                    <span className="min-slot-badge">Slot 1</span>
                    <h3>Frontal View (PA / AP)</h3>
                  </div>
                  {frontalImage && (
                    <button
                      className="min-btn-text"
                      onClick={() => frontalInputRef.current?.click()}
                    >
                      Change Image
                    </button>
                  )}
                </div>

                {!frontalImage ? (
                  <div
                    className="min-dropzone"
                    onClick={() => frontalInputRef.current?.click()}
                  >
                    <Upload className="min-drop-icon" size={36} />
                    <p className="min-drop-primary">Click to browse or drag Frontal Chest X-Ray</p>
                    <p className="min-drop-secondary">Supports PNG, JPG, JPEG, DICOM-exported formats</p>
                  </div>
                ) : (
                  <div className="min-preview-wrapper">
                    <img src={frontalImage} alt="Frontal Radiograph" className="min-preview-img" />

                    {/* Status Feedback */}
                    <div className="min-status-overlay">
                      {frontalStatus === 'validating' && (
                        <div className="min-status-badge status-loading">
                          <RefreshCw size={16} className="spin-icon" /> Checking modality & anatomy...
                        </div>
                      )}
                      {frontalStatus === 'valid' && (
                        <div className="min-status-badge status-success">
                          <CheckCircle2 size={16} /> {frontalMessage}
                        </div>
                      )}
                      {frontalStatus === 'invalid_not_xray' && (
                        <div className="min-status-badge status-danger">
                          <AlertOctagon size={16} /> {frontalMessage}
                        </div>
                      )}
                      {frontalStatus === 'invalid_not_chest' && (
                        <div className="min-status-badge status-warning">
                          <AlertTriangle size={16} /> {frontalMessage}
                        </div>
                      )}
                      {frontalStatus === 'invalid_wrong_view' && (
                        <div className="min-status-badge status-danger">
                          <AlertOctagon size={16} /> {frontalMessage}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Error Banner with Re-upload Action */}
                {(frontalStatus === 'invalid_not_xray' || frontalStatus === 'invalid_not_chest' || frontalStatus === 'invalid_wrong_view') && (
                  <div className="min-error-banner">
                    <div className="min-error-text">
                      <strong>
                        {frontalStatus === 'invalid_not_xray'
                          ? 'Modality Rejected'
                          : frontalStatus === 'invalid_not_chest'
                          ? 'Wrong Anatomical Region'
                          : 'Wrong Projection View'}
                      </strong>
                      <p>{frontalMessage}</p>
                    </div>
                    <button
                      className="min-btn-sm min-btn-danger"
                      onClick={() => frontalInputRef.current?.click()}
                    >
                      Upload New Image
                    </button>
                  </div>
                )}

                <input
                  type="file"
                  ref={frontalInputRef}
                  style={{ display: 'none' }}
                  accept="image/*"
                  onChange={(e) => handleFileChange(e, 'frontal')}
                />
              </div>

              {/* Box 2: Lateral Radiograph */}
              <div
                className={`min-upload-card ${
                  lateralStatus === 'valid'
                    ? 'card-valid'
                    : lateralStatus === 'invalid_not_xray' || lateralStatus === 'invalid_not_chest' || lateralStatus === 'invalid_wrong_view'
                    ? 'card-error'
                    : lateralStatus === 'validating'
                    ? 'card-validating'
                    : ''
                }`}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => handleDrop(e, 'lateral')}
              >
                <div className="min-card-top">
                  <div className="min-card-title">
                    <span className="min-slot-badge">Slot 2</span>
                    <h3>Lateral View</h3>
                  </div>
                  {lateralImage && (
                    <button
                      className="min-btn-text"
                      onClick={() => lateralInputRef.current?.click()}
                    >
                      Change Image
                    </button>
                  )}
                </div>

                {!lateralImage ? (
                  <div
                    className="min-dropzone"
                    onClick={() => lateralInputRef.current?.click()}
                  >
                    <Upload className="min-drop-icon" size={36} />
                    <p className="min-drop-primary">Click to browse or drag Lateral Chest X-Ray</p>
                    <p className="min-drop-secondary">Supports PNG, JPG, JPEG, DICOM-exported formats</p>
                  </div>
                ) : (
                  <div className="min-preview-wrapper">
                    <img src={lateralImage} alt="Lateral Radiograph" className="min-preview-img" />

                    {/* Status Feedback */}
                    <div className="min-status-overlay">
                      {lateralStatus === 'validating' && (
                        <div className="min-status-badge status-loading">
                          <RefreshCw size={16} className="spin-icon" /> Checking modality & anatomy...
                        </div>
                      )}
                      {lateralStatus === 'valid' && (
                        <div className="min-status-badge status-success">
                          <CheckCircle2 size={16} /> {lateralMessage}
                        </div>
                      )}
                      {lateralStatus === 'invalid_not_xray' && (
                        <div className="min-status-badge status-danger">
                          <AlertOctagon size={16} /> {lateralMessage}
                        </div>
                      )}
                      {lateralStatus === 'invalid_not_chest' && (
                        <div className="min-status-badge status-warning">
                          <AlertTriangle size={16} /> {lateralMessage}
                        </div>
                      )}
                      {lateralStatus === 'invalid_wrong_view' && (
                        <div className="min-status-badge status-danger">
                          <AlertOctagon size={16} /> {lateralMessage}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Error Banner with Re-upload Action */}
                {(lateralStatus === 'invalid_not_xray' || lateralStatus === 'invalid_not_chest' || lateralStatus === 'invalid_wrong_view') && (
                  <div className="min-error-banner">
                    <div className="min-error-text">
                      <strong>
                        {lateralStatus === 'invalid_not_xray'
                          ? 'Modality Rejected'
                          : lateralStatus === 'invalid_not_chest'
                          ? 'Wrong Anatomical Region'
                          : 'Wrong Projection View'}
                      </strong>
                      <p>{lateralMessage}</p>
                    </div>
                    <button
                      className="min-btn-sm min-btn-danger"
                      onClick={() => lateralInputRef.current?.click()}
                    >
                      Upload New Image
                    </button>
                  </div>
                )}

                <input
                  type="file"
                  ref={lateralInputRef}
                  style={{ display: 'none' }}
                  accept="image/*"
                  onChange={(e) => handleFileChange(e, 'lateral')}
                />
              </div>
            </div>

            {/* Smart Swap Banner if views are inverted */}
            {isSwapped && (
              <div className="min-swap-banner">
                <div className="swap-banner-info">
                  <AlertTriangle size={20} className="icon-amber" />
                  <div>
                    <strong>Projections Inverted</strong>
                    <p>The Lateral radiograph is in Slot 1 and the Frontal radiograph is in Slot 2.</p>
                  </div>
                </div>
                <button className="min-btn-swap" onClick={handleSwap}>
                  <RefreshCw size={14} /> Swap Images into Correct Slots
                </button>
              </div>
            )}



            {/* Action Button */}
            <div className="min-action-row">
              <button
                className={`min-primary-btn ${isReadyToAnalyze ? 'btn-ready' : 'btn-disabled'}`}
                disabled={!isReadyToAnalyze}
                onClick={handleRunAnalysis}
              >
                {isProcessing ? (
                  <>
                    <RefreshCw size={18} className="spin-icon" />
                    {processingStep === 1 && 'Step 1/3: Computing Visual Disease Probabilities...'}
                    {processingStep === 2 && 'Step 2/3: Generating LLM Clinical Report Draft...'}
                    {processingStep === 3 && 'Step 3/3: Cross-Checking & Removing Hallucinations...'}
                  </>
                ) : (
                  <>
                    <Activity size={18} /> Generate & Cross-Check Report
                  </>
                )}
              </button>
              {!isReadyToAnalyze && (
                <p className="min-action-hint">
                  {!frontalImage || !lateralImage
                    ? 'Please upload both Frontal and Lateral radiographs above to continue.'
                    : 'Both images must pass X-Ray modality and Chest anatomical validation to generate report.'}
                </p>
              )}
            </div>
          </div>
        ) : (
          /* ======================================================= */
          /* STAGE 2: RESULTS VIEW (PROBABILITIES -> LLM -> AUDIT)   */
          /* ======================================================= */
          <div className="min-results-section">
            {/* Top Bar with Thumbnails */}
            <div className="min-results-header">
              <div>
                <h2>Radiological Findings & Verified Report</h2>
                <p>Validated dual-view radiographs analyzed with RAD-DINO classifier and factual cross-checker.</p>
              </div>
              <div className="min-study-thumbnails">
                <div className="min-thumb-box">
                  <img src={resultData.frontal_image_url} alt="Frontal" />
                  <span>Frontal</span>
                </div>
                <div className="min-thumb-box">
                  <img src={resultData.lateral_image_url} alt="Lateral" />
                  <span>Lateral</span>
                </div>
              </div>
            </div>

            {/* SECTION 1: VISUAL CLASSIFIER PROBABILITIES */}
            <div className="min-card">
              <div className="min-card-header">
                <div className="min-header-left">
                  <Activity size={20} className="icon-cyan" />
                  <div>
                    <h3>1. Visual Classifier Probabilities</h3>
                    <p>Calculated disease risk across standard clinical chest pathologies.</p>
                  </div>
                </div>
                {resultData.all_probabilities && (
                  <button
                    className="min-btn-text"
                    onClick={() => setShowAll14(!showAll14)}
                  >
                    {showAll14 ? (
                      <>Hide Full 14-Finding List <ChevronUp size={14} /></>
                    ) : (
                      <>Show All 14 Findings <ChevronDown size={14} /></>
                    )}
                  </button>
                )}
              </div>

              {/* Detected High-Probability Findings */}
              <div className="min-findings-container">
                <div className="min-detected-row">
                  {resultData.detected_findings?.map((item, idx) => (
                    <div key={idx} className="min-finding-chip">
                      <span className="finding-name">{item.pathology}</span>
                      <span className={`finding-prob ${item.probability >= 40 ? 'prob-high' : 'prob-mid'}`}>
                        {item.probability}%
                      </span>
                    </div>
                  ))}
                </div>

                {/* Optional Expandable 14 CheXpert Pathologies */}
                {showAll14 && resultData.all_probabilities && (
                  <div className="min-all-probs-grid">
                    {Object.entries(resultData.all_probabilities).map(([pathology, prob]) => (
                      <div key={pathology} className="min-prob-row">
                        <div className="prob-label-row">
                          <span className="prob-label">{pathology}</span>
                          <span className={`prob-value ${prob >= 40 ? 'text-red' : prob >= 15 ? 'text-amber' : 'text-muted'}`}>
                            {prob}%
                          </span>
                        </div>
                        <div className="prob-bar-track">
                          <div
                            className={`prob-bar-fill ${prob >= 40 ? 'bar-red' : prob >= 15 ? 'bar-amber' : 'bar-normal'}`}
                            style={{ width: `${Math.min(100, Math.max(4, prob))}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* SECTION 2: INITIAL LLM REPORT DRAFT */}
            <div className="min-card">
              <div className="min-card-header">
                <div className="min-header-left">
                  <FileText size={20} className="icon-purple" />
                  <div>
                    <h3>2. Initial Report Draft (Multimodal LLM)</h3>
                    <p>Unverified narrative draft generated before cross-checking.</p>
                  </div>
                </div>
                <span className="min-tag tag-unverified">Pre-Verification Draft</span>
              </div>
              <div className="min-report-body draft-body">
                <p>{resultData.initial_report}</p>
              </div>
            </div>

            {/* SECTION 3: HALLUCINATION & FACTUAL AUDIT */}
            <div className="min-card min-card-audit">
              <div className="min-card-header">
                <div className="min-header-left">
                  <ShieldCheck size={20} className="icon-green" />
                  <div>
                    <h3>3. Hallucination Check & Clinical Verification</h3>
                    <p>Cross-references LLM statements against visual classifier probabilities.</p>
                  </div>
                </div>
                <span className={`min-tag ${resultData.is_concordant ? 'tag-concordant' : 'tag-adjusted'}`}>
                  {resultData.is_concordant ? '100% Concordant' : 'Factual Adjustments Made'}
                </span>
              </div>

              {/* Hallucinations Flagged & Removed */}
              {resultData.removed_diseases?.length > 0 && (
                <div className="min-audit-box audit-danger">
                  <div className="audit-box-title">
                    <ShieldAlert size={18} />
                    <span>Hallucinations Flagged & Removed ({resultData.removed_diseases.length})</span>
                  </div>
                  <div className="audit-items">
                    {resultData.removed_diseases.map((h, i) => (
                      <div key={i} className="audit-item">
                        <div className="audit-item-top">
                          <strong className="text-red">Claimed Pathology: {h.pathology}</strong>
                          <span className="audit-prob-badge">Visual Classifier Evidence: {h.probability}%</span>
                        </div>
                        <p className="audit-reason">{h.reason}</p>
                        {h.original_snippet && (
                          <div className="audit-snippet">
                            <span>Draft sentence removed:</span> "{h.original_snippet}"
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Omissions Detected & Added */}
              {resultData.omitted_diseases_added?.length > 0 && (
                <div className="min-audit-box audit-warning">
                  <div className="audit-box-title">
                    <AlertTriangle size={18} />
                    <span>Omissions Detected & Added ({resultData.omitted_diseases_added.length})</span>
                  </div>
                  <div className="audit-items">
                    {resultData.omitted_diseases_added.map((o, i) => (
                      <div key={i} className="audit-item">
                        <div className="audit-item-top">
                          <strong className="text-amber">Missed Pathology: {o.pathology}</strong>
                          <span className="audit-prob-badge">Visual Classifier Risk: {o.probability}%</span>
                        </div>
                        <p className="audit-reason">{o.action}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* If 100% concordant */}
              {resultData.is_concordant && (
                <div className="min-audit-box audit-success">
                  <div className="audit-box-title">
                    <CheckCircle2 size={18} />
                    <span>No Hallucinations Detected</span>
                  </div>
                  <p className="audit-success-msg">
                    All findings asserted in the report are fully corroborated by the visual classifier probabilities.
                  </p>
                </div>
              )}

              {/* FINAL VERIFIED REPORT */}
              <div className="min-final-report">
                <div className="final-report-top">
                  <div>
                    <h4 className="final-title">Final Verified Clinical Report</h4>
                    <span className="final-stamp">Factual & Verified by Dual-Stage Cross-Checker</span>
                  </div>
                  <button className="min-btn-copy" onClick={copyReport}>
                    {copied ? <><Check size={14} /> Copied</> : <><Copy size={14} /> Copy Report</>}
                  </button>
                </div>
                <div className="final-report-text">
                  <p>{resultData.final_verified_report}</p>
                </div>
              </div>
            </div>

            {/* Bottom Actions */}
            <div className="min-bottom-actions">
              <button className="min-btn-outline" onClick={handleReset}>
                <RefreshCw size={15} /> Analyze Another Radiograph Study
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
