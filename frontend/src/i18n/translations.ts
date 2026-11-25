export type Language = "en" | "hi";

export interface TranslationDictionary {
  // Navigation
  brand: string;
  navOverview: string;
  navTraffic: string;
  navDetection: string;
  navExperiments: string;
  navResearch: string;
  navAbout: string;

  // Themes
  themeLight: string;
  themeDark: string;
  themeHighContrast: string;
  themeCyberpunk: string;

  // Status
  statusBackendReady: string;
  statusBackendOffline: string;
  statusDegraded: string;
  statusModelLoaded: string;
  statusModelNotLoaded: string;
  statusConnecting: string;

  // Overview Page
  overviewTitle: string;
  overviewSubtitle: string;
  kpiTestAccuracy: string;
  kpiMacroF1: string;
  kpiAttackRecall: string;
  kpiFalsePositiveRate: string;
  kpiQuantumQubits: string;
  kpiTotalParameters: string;
  activeModelLabel: string;
  datasetLabel: string;
  pipelineDiagramTitle: string;
  historicalNoticeTitle: string;
  historicalNoticeBody: string;

  // Traffic Analysis Page
  trafficTitle: string;
  trafficSubtitle: string;
  dragDropTitle: string;
  dragDropSubtitle: string;
  loadSampleBtn: string;
  runBatchBtn: string;
  downloadCsvBtn: string;
  detectedRows: string;
  detectedFeatures: string;
  validationPassed: string;
  validationFailed: string;
  attackDetected: string;
  benignDetected: string;
  confidenceScore: string;
  uploadedNotice: string;

  // Detection Lab Page
  labTitle: string;
  labSubtitle: string;
  presetSelectorLabel: string;
  selectPreset: string;
  predictActionBtn: string;
  predictionResultTitle: string;
  predictedClassLabel: string;
  attackProbLabel: string;
  benignProbLabel: string;
  quantumInspectionTitle: string;
  quantumAnglesLabel: string;
  pauliZExpectationsLabel: string;
  decisionSupportDisclaimer: string;

  // Experiments Page
  experimentsTitle: string;
  experimentsSubtitle: string;
  architectureCol: string;
  paramsCol: string;
  accuracyCol: string;
  f1Col: string;
  recallCol: string;
  fprCol: string;
  decisionThresholdCol: string;
  trainingTimeCol: string;
  methodologyNoteTitle: string;
  methodologyNoteBody: string;

  // Research Page
  researchTitle: string;
  researchSubtitle: string;
  researchAbstractTitle: string;
  researchAbstractBody: string;
  hardwareDisclaimerTitle: string;
  hardwareDisclaimerBody: string;
  downloadPaperBtn: string;
  downloadStatsBtn: string;

  // About Page
  aboutTitle: string;
  aboutSubtitle: string;
  techStackTitle: string;
  reproducibilityTitle: string;
  endpointsTitle: string;
  responsibleUseTitle: string;
  responsibleUseBody: string;

  // Common UI
  loading: string;
  error: string;
  reset: string;
  viewAll: string;
  close: string;
}

export const translations: Record<Language, TranslationDictionary> = {
  en: {
    brand: "Q-NIDS",
    navOverview: "Overview",
    navTraffic: "Traffic Analysis",
    navDetection: "Detection Lab",
    navExperiments: "Experiments",
    navResearch: "Research",
    navAbout: "About & System",

    themeLight: "Light Research",
    themeDark: "Dark Research",
    themeHighContrast: "High Contrast",
    themeCyberpunk: "Cyberpunk",

    statusBackendReady: "API Online",
    statusBackendOffline: "API Offline",
    statusDegraded: "Degraded",
    statusModelLoaded: "Model Active",
    statusModelNotLoaded: "Model Inactive",
    statusConnecting: "Connecting...",

    overviewTitle: "Hybrid Classical–Quantum Intrusion Detection",
    overviewSubtitle: "Empirical evaluation of Variational Quantum Circuits and Quantum Data Re-uploading on the UNSW-NB15 benchmark",
    kpiTestAccuracy: "Test Accuracy",
    kpiMacroF1: "Macro F1",
    kpiAttackRecall: "Attack Recall (DR)",
    kpiFalsePositiveRate: "False Positive Rate (FPR)",
    kpiQuantumQubits: "Simulated Qubits",
    kpiTotalParameters: "Trainable Parameters",
    activeModelLabel: "Deployed Model",
    datasetLabel: "Evaluated Dataset",
    pipelineDiagramTitle: "Inference & Zero-Leakage Architecture",
    historicalNoticeTitle: "Verified Benchmark Metrics",
    historicalNoticeBody: "Metrics shown below represent verified empirical results on the held-out UNSW-NB15 test partition across 3 random seeds. They are not simulated synthetic telemetry.",

    trafficTitle: "Network Traffic Flow Inspector",
    trafficSubtitle: "Batch CSV traffic validation and real-time inference using pre-fitted zero-leakage preprocessing",
    dragDropTitle: "Drag & drop network flow CSV here",
    dragDropSubtitle: "Strict 39-feature CIC-IoT-2023 schema required. Missing values will be validated.",
    loadSampleBtn: "Load 25 Valid Test Flows",
    runBatchBtn: "Run Batch Classification",
    downloadCsvBtn: "Download Classified CSV",
    detectedRows: "Total Rows Detected",
    detectedFeatures: "Valid Features Matched",
    validationPassed: "Schema Validation Passed",
    validationFailed: "Validation Failed",
    attackDetected: "Attacks Detected",
    benignDetected: "Benign Flows",
    confidenceScore: "Confidence Score",
    uploadedNotice: "Predictions generated by model inference are decision-support classifications, not autonomous firewall policy executions.",

    labTitle: "Single-Flow Detection Laboratory",
    labSubtitle: "Interactive feature perturbation, real-time prediction, and quantum expectation state inspection",
    presetSelectorLabel: "Load CIC-IoT-2023 Held-Out Flow Preset",
    selectPreset: "-- Select a real network flow preset --",
    predictActionBtn: "Execute Model Inference",
    predictionResultTitle: "Inference Diagnostics",
    predictedClassLabel: "Classification Outcome",
    attackProbLabel: "Attack Probability",
    benignProbLabel: "Benign Probability",
    quantumInspectionTitle: "Quantum Circuit State Inspection",
    quantumAnglesLabel: "Classical Feature Rotation Angles θ ∈ [-π, π]",
    pauliZExpectationsLabel: "Measured Pauli-Z Expectation Values ⟨Z_i⟩ ∈ [-1, 1]",
    decisionSupportDisclaimer: "This system is an operational decision-support tool. It does not automatically quarantine network ports or substitute for firewall rules.",

    experimentsTitle: "Controlled Research Benchmarks",
    experimentsSubtitle: "Strict 6-model comparison across 3 fixed random seeds (42, 123, 999) under identical splits",
    architectureCol: "Model Architecture",
    paramsCol: "Parameters",
    accuracyCol: "Test Accuracy",
    f1Col: "Macro F1",
    recallCol: "Attack Recall",
    fprCol: "FPR (False Alarms)",
    decisionThresholdCol: "Tuned Cutoff (τ*)",
    trainingTimeCol: "Training Time (s)",
    methodologyNoteTitle: "Controlled Methodology Protocol",
    methodologyNoteBody: "All models share identical stratified splits (70% train, 15% validation, 15% test). Preprocessors were fitted exclusively on training data. Optimal decision thresholds (τ*) were tuned on validation probabilities using Youden's J statistic, and checkpoints were selected on minimum validation loss.",

    researchTitle: "Methodology & Empirical Research",
    researchSubtitle: "Formal investigation into Fourier frequency expressivity in Quantum Data Re-uploading for tabular network intrusion detection",
    researchAbstractTitle: "Abstract & Scientific Context",
    researchAbstractBody: "Variational quantum circuits with single data encoding are mathematically constrained to low-degree Fourier series. Interleaved quantum data re-uploading expands the accessible spectrum, enabling a 1,113-parameter hybrid model to match a 6,401-parameter classical MLP baseline.",
    hardwareDisclaimerTitle: "Simulator Hardware Disclaimer",
    hardwareDisclaimerBody: "All quantum circuits were executed via PennyLane statevector simulation on classical CPU hardware (default.qubit). No physical NISQ hardware was utilized. Competitive parameter efficiency does not constitute quantum advantage.",
    downloadPaperBtn: "Read Research Report (MD)",
    downloadStatsBtn: "Download Aggregate JSON",

    aboutTitle: "System Architecture & Verification",
    aboutSubtitle: "System environment, dependency versions, API contracts, and reproducibility specifications",
    techStackTitle: "Core Technologies",
    reproducibilityTitle: "Zero-Leakage & Reproducibility Guarantee",
    endpointsTitle: "Production REST API Endpoints",
    responsibleUseTitle: "Responsible Cybersecurity Disclosure",
    responsibleUseBody: "This system is intended for research analysis and Security Operations Center decision support. It must not be deployed as an autonomous blocking mechanism without human verification.",

    loading: "Processing request...",
    error: "An error occurred",
    reset: "Reset Form",
    viewAll: "View All",
    close: "Close",
  },

  hi: {
    brand: "Q-NIDS",
    navOverview: "अवलोकन (Overview)",
    navTraffic: "ट्रैफ़िक विश्लेषण (Traffic)",
    navDetection: "डिटेक्शन लैब (Detection Lab)",
    navExperiments: "प्रयोग एवं परिणाम (Experiments)",
    navResearch: "अनुसंधान (Research)",
    navAbout: "सिस्टम विवरण (About)",

    themeLight: "लाइट रिसर्च (Light)",
    themeDark: "डार्क रिसर्च (Dark)",
    themeHighContrast: "हाई कंट्रास्ट (High Contrast)",
    themeCyberpunk: "साइबरपंक (Cyberpunk)",

    statusBackendReady: "एपीआई ऑनलाइन",
    statusBackendOffline: "एपीआई ऑफलाइन",
    statusDegraded: "सीमित (Degraded)",
    statusModelLoaded: "मॉडल सक्रिय",
    statusModelNotLoaded: "मॉडल निष्क्रिय",
    statusConnecting: "कनेक्ट हो रहा है...",

    overviewTitle: "हाइब्रिड क्लासिकल–क्वांटम घुसपैठ पहचान प्रणाली",
    overviewSubtitle: "UNSW-NB15 नेटवर्क डेटासेट पर वेरिएशनल क्वांटम सर्किट और क्वांटम डेटा री-अपलोडिंग का अनुभवजन्य मूल्यांकन",
    kpiTestAccuracy: "परीक्षण सटीकता (Accuracy)",
    kpiMacroF1: "मैक्रो F1 स्कोर",
    kpiAttackRecall: "अटैक रिकॉल (Detection Rate)",
    kpiFalsePositiveRate: "मिथ्या चेतावनी दर (FPR)",
    kpiQuantumQubits: "सिम्युलेटेड क्यूबिट्स",
    kpiTotalParameters: "प्रशिक्षण योग्य पैरामीटर",
    activeModelLabel: "सक्रिय मॉडल",
    datasetLabel: "परीक्षित डेटासेट",
    pipelineDiagramTitle: "अनुमान एवं शून्य-लीकेज पाइपलाइन",
    historicalNoticeTitle: "सत्यापित बेंचमार्क परिणाम",
    historicalNoticeBody: "नीचे दिए गए मेट्रिक्स 3 रैंडम बीजों (Seeds) पर UNSW-NB15 टेस्ट स्प्लिट के वास्तविक परिणाम हैं। यह कोई काल्पनिक या नकली डेटा नहीं है।",

    trafficTitle: "नेटवर्क फ़्लो ट्रैफ़िक विश्लेषक",
    trafficSubtitle: "शून्य-लीकेज प्रीप्रोसेसिंग के साथ बैच CSV ट्रैफ़िक सत्यापन और मॉडल अनुमान",
    dragDropTitle: "नेटवर्क फ़्लो CSV यहाँ ड्रैग करें",
    dragDropSubtitle: "CIC-IoT-2023 का 39-फ़ीचर प्रारूप आवश्यक है। रिक्त मानों की स्वचालित जाँच होगी।",
    loadSampleBtn: "25 मान्य टेस्ट फ़्लो लोड करें",
    runBatchBtn: "बैच वर्गीकरण चलाएँ",
    downloadCsvBtn: "परिणाम CSV डाउनलोड करें",
    detectedRows: "कुल प्राप्त पंक्तियाँ",
    detectedFeatures: "सत्यापित फ़ीचर्स",
    validationPassed: "स्कीमा सत्यापन सफल",
    validationFailed: "सत्यापन विफल",
    attackDetected: "पहचाने गए साइबर हमले",
    benignDetected: "सामान्य ट्रैफ़िक फ़्लो",
    confidenceScore: "सटीकता विश्वास (Confidence)",
    uploadedNotice: "मॉडल द्वारा दिए गए परिणाम निर्णय-सहायता वर्गीकरण हैं, यह स्वचालित फ़ायरवॉल नियम नहीं हैं।",

    labTitle: "सिंगल-फ़्लो डिटेक्शन प्रयोगशाला",
    labSubtitle: "इंटरैक्टिव फ़ीचर परिवर्तन, वास्तविक समय अनुमान और क्वांटम अवस्था निरीक्षण",
    presetSelectorLabel: "CIC-IoT-2023 फ़्लो प्रीसेट चुनें",
    selectPreset: "-- वास्तविक नेटवर्क फ़्लो प्रीसेट चुनें --",
    predictActionBtn: "मॉडल अनुमान चलाएँ",
    predictionResultTitle: "अनुमान विश्लेषण परिणाम",
    predictedClassLabel: "वर्गीकरण परिणाम",
    attackProbLabel: "हमले की संभावना (Attack Prob)",
    benignProbLabel: "सामान्य संभावना (Benign Prob)",
    quantumInspectionTitle: "क्वांटम सर्किट अवस्था निरीक्षण",
    quantumAnglesLabel: "क्लासिकल फ़ीचर रोटेशन कोण θ ∈ [-π, π]",
    pauliZExpectationsLabel: "मापे गए पॉली-Z प्रत्याशा मान ⟨Z_i⟩ ∈ [-1, 1]",
    decisionSupportDisclaimer: "यह प्रणाली सुरक्षा विश्लेषक के निर्णय में सहायता के लिए है। यह फ़ायरवॉल का पूर्ण विकल्प नहीं है।",

    experimentsTitle: "नियंत्रित अनुसंधान अध्ययन एवं बेंचमार्क",
    experimentsSubtitle: "3 स्वतंत्र बीजों (42, 123, 999) पर 6 मॉडलों का नियंत्रित तुलनात्मक अध्ययन",
    architectureCol: "मॉडल वास्तुकला",
    paramsCol: "पैरामीटर्स",
    accuracyCol: "सटीकता (Accuracy)",
    f1Col: "मैक्रो F1",
    recallCol: "अटैक रिकॉल",
    fprCol: "फ़ॉल्स पॉजिटिव दर",
    decisionThresholdCol: "ट्यून किया गया कटऑफ (τ*)",
    trainingTimeCol: "प्रशिक्षण समय (सेकंड)",
    methodologyNoteTitle: "कठोर अनुसंधान कार्यप्रणाली",
    methodologyNoteBody: "सभी मॉडलों ने समान स्प्लिट्स (70% ट्रेन, 15% वैलिडेशन, 15% टेस्ट) का उपयोग किया। निर्णय थ्रेशोल्ड (τ*) वैलिडेशन डेटा पर यूडन जे (Youden's J) द्वारा अनुकूलित किए गए, और चेकपॉइंट न्यूनतम वैलिडेशन लॉस पर चुने गए।",

    researchTitle: "अनुसंधान कार्यप्रणाली एवं सिद्धांत",
    researchSubtitle: "नेटवर्क घुसपैठ पहचान में क्वांटम डेटा री-अपलोडिंग की फूरियर आवृत्ति और अभिव्यक्ति क्षमता का विश्लेषण",
    researchAbstractTitle: "वैज्ञानिक सारांश (Abstract)",
    researchAbstractBody: "एकल एन्कोडिंग वाले वेरिएशनल क्वांटम सर्किट सीमित फूरियर डिग्री तक सीमित रहते हैं। डेटा री-अपलोडिंग इस सीमा को पार करती है, जिससे 1,113-पैरामीटर का मॉडल 6,401-पैरामीटर के क्लासिकल MLP के समान सटीकता प्राप्त करता है।",
    hardwareDisclaimerTitle: "सिम्युलेटर हार्डवेयर सूचना",
    hardwareDisclaimerBody: "सभी क्वांटम सर्किट PennyLane CPU सिम्युलेटर (default.qubit) पर चलाए गए हैं। किसी भौतिक क्वांटम हार्डवेयर का उपयोग नहीं हुआ है। कम पैरामीटर होना क्वांटम श्रेष्ठता का प्रमाण नहीं है।",
    downloadPaperBtn: "अनुसंधान रिपोर्ट पढ़ें (MD)",
    downloadStatsBtn: "सांख्यिकी JSON डाउनलोड करें",

    aboutTitle: "सिस्टम वास्तुकला और तकनीकी जानकारी",
    aboutSubtitle: "पर्यावरण, निर्भरता संस्करण, एपीआई अनुबंध और पुनरुत्पादन विनिर्देश",
    techStackTitle: "मूल प्रौद्योगिकियाँ (Tech Stack)",
    reproducibilityTitle: "शून्य-लीकेज एवं पुनरुत्पादन गारंटी",
    endpointsTitle: "उत्पादन REST एपीआई एंडपॉइंट्स",
    responsibleUseTitle: "जिम्मेदार उपयोग की अस्वीकरण सूचना",
    responsibleUseBody: "यह प्रणाली अनुसंधान और सुरक्षा संचालन केंद्र (SOC) निर्णय समर्थन के लिए है। मानव सत्यापन के बिना इसे स्वचालित ब्लॉक करने के लिए तैनात न करें।",

    loading: "प्रक्रिया जारी है...",
    error: "एक त्रुटि हुई",
    reset: "रीसेट करें",
    viewAll: "सभी देखें",
    close: "बंद करें",
  },
};
