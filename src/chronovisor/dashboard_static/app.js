const els = {
  capture: document.getElementById("capture-value"),
  captureSub: document.getElementById("capture-sub"),
  eventFeed: document.getElementById("event-feed"),
  healthCaption: document.getElementById("health-caption"),
  healthCapture: document.getElementById("health-capture"),
  healthExit: document.getElementById("health-exit"),
  healthFailed: document.getElementById("health-failed"),
  healthQueued: document.getElementById("health-queued"),
  healthRetry: document.getElementById("health-retry"),
  healthRunning: document.getElementById("health-running"),
  knowledgeBars: document.getElementById("knowledge-bars"),
  knowledgeCaption: document.getElementById("knowledge-caption"),
  knowledgeCategories: document.getElementById("knowledge-categories"),
  knowledgeChart: document.getElementById("knowledge-chart"),
  knowledgeModeButtons: document.querySelectorAll("[data-knowledge-mode]"),
  knowledgePages: document.getElementById("knowledge-pages"),
  knowledgeSize: document.getElementById("knowledge-size"),
  knowledgeTotal: document.getElementById("knowledge-total"),
  lanShareButton: document.getElementById("lan-share-button"),
  pages: document.getElementById("pages-value"),
  pagesSub: document.getElementById("pages-sub"),
  pendingChart: document.getElementById("pending-chart"),
  pendingCurrent: document.getElementById("pending-current"),
  queue: document.getElementById("queue-value"),
  queueSub: document.getElementById("queue-sub"),
  saveDetail: document.getElementById("save-detail"),
  saveFeed: document.getElementById("save-feed"),
  saveHeatmap: document.getElementById("save-heatmap"),
  saveLoadTooltip: document.getElementById("save-load-tooltip"),
  saveModeButtons: document.querySelectorAll("[data-save-mode]"),
  saveMonths: document.getElementById("save-months"),
  saveTotal: document.getElementById("save-total"),
  saves: document.getElementById("saves-value"),
  services: document.getElementById("services-value"),
  servicesSub: document.getElementById("services-sub"),
  statePill: document.getElementById("state-pill"),
  stateText: document.getElementById("state-text"),
  trendCaption: document.getElementById("trend-caption"),
};

let saveHistoryMode = "daily";
let latestSaveHistory = null;
let selectedSaveDate = null;
let knowledgeMixMode = "size";
let latestKnowledgeMix = null;
let saveLoadHitRegions = [];

const KNOWLEDGE_COLORS = [
  "#828fff",
  "#3dd68c",
  "#e8b04b",
  "#c983f7",
  "#ff7e79",
  "#56a8ff",
  "#e0cd6d",
  "#4fc8b4",
];
