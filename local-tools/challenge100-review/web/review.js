"use strict";

const categoryLabels = {
  acrobatics: "翻转与杂技", hand_support: "手／肘支撑", floor_recovery: "地面动作与起身",
  seated: "坐姿与坐立转换", terrain: "楼梯／梯子／高差障碍", object_interaction: "物体交互",
  agility_sport: "高速与运动", dance: "舞蹈", rare_actions: "罕见动作"
};
const controls = Object.fromEntries(["search", "category", "retention", "original"].map(id => [id, document.getElementById(id)]));
let reviewMotions = [];

function reviewElement(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function checkedMotions(parsed) {
  if (!Array.isArray(parsed) || parsed.length !== 100) throw new Error("选片清单必须包含完整 100 段。");
  const names = new Set();
  for (const motion of parsed) {
    if (!motion || typeof motion.filename !== "string" || !/^[A-Za-z0-9_-]+$/.test(motion.filename)) throw new Error("选片标识格式无效。");
    if (names.has(motion.filename)) throw new Error("选片清单存在重复动作。");
    if (!Number.isInteger(motion.frames) || motion.frames < 1 || !Number.isFinite(motion.fps) || motion.fps <= 0) throw new Error("选片帧数或帧率无效。");
    if (!Number.isInteger(motion.review_index) || motion.review_index < 1 || motion.review_index > 100) throw new Error("选片序号无效。");
    names.add(motion.filename);
  }
  if (parsed.filter(motion => motion.original11 === true).length !== 11) throw new Error("清单未完整保留原来的 11 段。");
  return parsed;
}

async function fetchReviewJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`审阅数据加载失败：${path}，HTTP ${response.status}`);
  return response.json();
}

async function motionsWithTraining() {
  const [selection, retention] = await Promise.all([
    fetchReviewJson("selection/challenge100-proposal.json"), fetchReviewJson("selection/training-retention.json")
  ]);
  const motions = checkedMotions(selection);
  if (!retention || !retention.motions || retention.selected_count !== motions.length) throw new Error("清洗标注与选片清单不一致。");
  for (const motion of motions) {
    const decision = retention.motions[motion.filename];
    if (!decision || typeof decision.keep !== "boolean" || typeof decision.reason !== "string") throw new Error("缺少有效的逐片段清洗标注。");
    if (!decision.keep && !decision.reason) throw new Error("被剔除的片段缺少原因。");
    motion.training_filter = decision;
  }
  return motions;
}

function trainingStatus(motion) {
  const filter = motion.training_filter;
  if (filter.keep) return { key: "kept", text: "训练保留" };
  return { key: "rejected", text: `训练剔除：${filter.reason}` };
}

function activeFilters() {
  return {
    query: controls.search.value.trim().toLowerCase(), category: controls.category.value,
    retention: controls.retention.value, original: controls.original.checked
  };
}

function matchesFilters(motion, filters) {
  const searchable = [motion.filename, motion.family, motion.content_props, motion.content_short_description,
    motion.content_technical_description, motion.content_natural_desc_1, categoryLabels[motion.bucket]].join(" ").toLowerCase();
  return (!filters.query || searchable.includes(filters.query)) && (!filters.category || motion.bucket === filters.category)
    && (!filters.retention || trainingStatus(motion).key === filters.retention) && (!filters.original || motion.original11 === true);
}

function pauseOtherVideos(event) {
  for (const video of document.querySelectorAll("video")) {
    if (video !== event.target) video.pause();
  }
}

function motionVideo(motion, card) {
  const video = reviewElement("video");
  video.controls = true;
  video.playsInline = true;
  video.preload = "none";
  video.setAttribute("aria-label", `${motion.filename} 的三角色同步视频`);
  card.addEventListener("toggle", () => {
    if (!card.open) { video.pause(); return; }
    if (video.getAttribute("src")) return;
    video.src = `videos/${encodeURIComponent(motion.filename)}.mp4`;
    video.poster = `previews/${encodeURIComponent(motion.filename)}.png`;
  });
  video.addEventListener("play", pauseOtherVideos);
  video.addEventListener("error", () => {
    card.append(reviewElement("p", "该视频尚未就绪或加载失败；请核对渲染状态后重试。", "video-error"));
  }, { once: true });
  return video;
}

function motionLinks(motion) {
  const links = reviewElement("div", undefined, "motion-links");
  const videoLink = reviewElement("a", "下载本段完整视频");
  videoLink.href = `videos/${encodeURIComponent(motion.filename)}.mp4`;
  videoLink.download = "";
  const receiptLink = reviewElement("a", "输入／相机／显示变换记录");
  receiptLink.href = `receipts/${encodeURIComponent(motion.filename)}.json`;
  links.append(videoLink, receiptLink);
  return links;
}

function motionSummary(motion) {
  const summary = reviewElement("summary");
  summary.append(reviewElement("span", `${String(motion.review_index).padStart(3, "0")} · ${motion.filename}`, "motion-title"));
  summary.append(reviewElement("span", categoryLabels[motion.bucket] || motion.bucket, "badge"));
  if (motion.original11) summary.append(reviewElement("span", "原 11 段", "badge"));
  const status = trainingStatus(motion);
  summary.append(reviewElement("span", status.text, `badge ${status.key === "kept" ? "retained" : status.key}`));
  return summary;
}

function sceneWarnings(motion) {
  const props = String(motion.content_props || "").trim();
  const involvesScene = motion.bucket === "terrain" || motion.bucket === "object_interaction" || (props && props !== "0");
  if (!involvesScene) return [];
  return [reviewElement("p", "场景边界：物体／地形仅有动作语义，源场景几何未提供。视频显示统一平地，不能据此将踩高处或坐物体判为悬空失败。", "motion-context")];
}

function motionCard(motion) {
  const card = reviewElement("details");
  card.id = motion.filename;
  card.dataset.filename = motion.filename;
  const seconds = ((motion.frames - 1) / motion.fps).toFixed(2);
  const description = motion.content_short_description || motion.content_technical_description || motion.family;
  const context = `${motion.frames} 个源帧 · ${motion.fps} Hz · 端点跨度 ${seconds} s · 官方类别：${motion.category || "未注明"} · 道具标签：${motion.content_props || "未注明"}`;
  card.append(motionSummary(motion), reviewElement("p", description, "motion-description"),
    reviewElement("p", context, "motion-context"), ...sceneWarnings(motion), motionVideo(motion, card), motionLinks(motion));
  return card;
}

function updateVisibility() {
  const filters = activeFilters();
  let shown = 0;
  for (const motion of reviewMotions) {
    const card = document.getElementById(motion.filename);
    card.hidden = !matchesFilters(motion, filters);
    if (!card.hidden) shown += 1;
    if (card.hidden) card.querySelector("video").pause();
  }
  document.getElementById("selection-status").textContent = `显示 ${shown} / ${reviewMotions.length} 段。训练剔除并不等于重定向失败；详见实际清洗流程。`;
}

function populateCategories() {
  const categories = [...new Set(reviewMotions.map(motion => motion.bucket))];
  for (const category of categories) {
    const option = reviewElement("option", categoryLabels[category] || category);
    option.value = category;
    controls.category.append(option);
  }
}

async function initializeReview() {
  reviewMotions = await motionsWithTraining();
  document.getElementById("motions").append(...reviewMotions.map(motionCard));
  populateCategories();
  for (const control of Object.values(controls)) control.addEventListener("input", updateVisibility);
  updateVisibility();
  const requested = location.hash.slice(1);
  const selected = reviewMotions.find(motion => motion.filename === requested) || reviewMotions[0];
  const first = document.getElementById(selected.filename);
  first.open = true;
}

initializeReview().catch(error => {
  document.getElementById("selection-status").textContent = error.message;
});
