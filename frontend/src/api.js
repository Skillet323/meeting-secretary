import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  timeout: 120000,
});

export const uploadMeeting = async (formData) => {
  const response = await api.post("/upload_meeting", formData);
  return response.data;
};

export const getMeeting = async (id) => {
  const response = await api.get(`/meeting/${id}`);
  return response.data;
};

export const getMeetingProgress = async (id) => {
  const response = await api.get(`/meeting/${id}/progress`);
  return response.data;
};

export const exportMeeting = async (id, format) => {
  const response = await api.get(`/meeting/${id}/export?format=${format}`, {
    responseType: "blob",
  });
  return response.data;
};

export const getMetrics = async (limit = 10) => {
  const response = await api.get(`/metrics?limit=${limit}`);
  return response.data;
};

export const getEvaluations = async (limit = 10) => {
  const response = await api.get(`/evaluations?limit=${limit}`);
  return response.data;
};