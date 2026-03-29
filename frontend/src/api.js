import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const uploadMeeting = async (formData) => {
  const response = await axios.post(`${API_BASE}/upload_meeting`, formData);
  return response.data;
};

export const getMeeting = async (id) => {
  const response = await axios.get(`${API_BASE}/meeting/${id}`);
  return response.data;
};

export const getMeetingProgress = async (id) => {
  const response = await axios.get(`${API_BASE}/meeting/${id}/progress`);
  return response.data;
};

export const exportMeeting = async (id, format) => {
  const response = await axios.get(`${API_BASE}/meeting/${id}/export?format=${format}`, {
    responseType: "blob",
  });
  return response.data;
};

export const getRecentMetrics = async (limit = 10) => {
  const response = await axios.get(`${API_BASE}/metrics?limit=${limit}`);
  return response.data;
};

export const getEvaluations = async (limit = 10) => {
  const response = await axios.get(`${API_BASE}/evaluations?limit=${limit}`);
  return response.data;
};
