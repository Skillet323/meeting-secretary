import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  timeout: 600000,
});

export const uploadMeeting = async (formData, onProgress = null) => {
  const response = await api.post("/upload_meeting", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(percentCompleted);
      }
    },
  });
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

export const getRecentMetrics = async (limit = 10) => {
  const response = await api.get(`/metrics?limit=${limit}`);
  return response.data;
};

export const getMeetings = async (limit = 10) => {
  const response = await api.get(`/meetings?limit=${limit}`);
  return response.data;
};

export const getEvaluations = async (limit = 10) => {
  const response = await api.get(`/evaluations?limit=${limit}`);
  return response.data;
};
export const evaluateMeeting = async (id) => {
  const response = await api.post(`/evaluate/meeting/${id}`);
  return response.data;
};

export const getEvaluationDetails = async (runId) => {
  const response = await api.get(`/evaluation/${runId}`);
  return response.data;
};

export const getStats = async () => {
  const response = await api.get("/stats");
  return response.data;
};

export const getParticipants = async () => {
  const response = await api.get("/participants");
  return response.data;
};

