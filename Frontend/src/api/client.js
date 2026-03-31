const API_BASE = 'http://localhost:8000/api/v1';

/**
 * Fetch 3 speaking questions (1 static + 2 dynamic)
 */
export async function fetchSpeakingQuestions() {
  const res = await fetch(`${API_BASE}/speaking/questions`);
  if (!res.ok) throw new Error('Failed to fetch speaking questions');
  const data = await res.json();
  return data.questions;
}

/**
 * Submit a speaking response for evaluation
 * @param {Blob} audioBlob  WAV audio blob
 * @param {string} question The question that was asked
 */
export async function submitSpeakingResponse(audioBlob, question) {
  const form = new FormData();
  form.append('audio', audioBlob, 'recording.wav');
  form.append('question', question);

  const res = await fetch(`${API_BASE}/evaluate`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) throw new Error('Speaking evaluation failed');
  return res.json();
}

/**
 * Fetch listening clips (4 clips with audio)
 */
export async function fetchListeningClips() {
  const res = await fetch(`${API_BASE}/listening/clips`);
  if (!res.ok) throw new Error('Failed to fetch listening clips');
  const data = await res.json();
  return data.clips; // { session_id, clips: [...] }
}

/**
 * Submit a listening response
 * @param {Blob} audioBlob
 * @param {string} sessionId
 * @param {string} clipId
 * @param {number} questionIndex
 */
export async function submitListeningResponse(audioBlob, sessionId, clipId, questionIndex = 0) {
  const form = new FormData();
  form.append('audio', audioBlob, 'recording.wav');
  form.append('session_id', sessionId);
  form.append('clip_id', clipId);
  form.append('question_index', questionIndex.toString());

  const res = await fetch(`${API_BASE}/listening/respond`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) throw new Error('Listening response submission failed');
  return res.json();
}

/**
 * Aggregate listening results
 * @param {Array} clipResults
 */
export async function aggregateListeningResults(clipResults) {
  const res = await fetch(`${API_BASE}/listening/aggregate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(clipResults),
  });
  if (!res.ok) throw new Error('Listening aggregation failed');
  return res.json();
}
