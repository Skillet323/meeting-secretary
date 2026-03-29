import React, { useEffect, useMemo, useState } from "react";
import {
  Container,
  Box,
  Typography,
  Paper,
  Button,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Snackbar,
  Alert,
  Divider,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  Tabs,
  Tab,
  Chip,
} from "@mui/material";
import {
  Upload,
  Download,
  Assessment,
  Person,
  Timeline,
  Refresh,
  BarChart as BarChartIcon,
} from "@mui/icons-material";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { format } from "date-fns";

import {
  uploadMeeting,
  getMeeting,
  getMeetingProgress,
  exportMeeting,
  getRecentMetrics,
  getEvaluations,
} from "./api";

function safeNumber(value, digits = 1) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "—";
  return num.toFixed(digits);
}

function taskLabel(task) {
  return task?.description || "Untitled task";
}

export default function App() {
  const [meetings, setMeetings] = useState([]);
  const [selectedMeeting, setSelectedMeeting] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("transcript");
  const [processingStatus, setProcessingStatus] = useState("");
  const [recentMetrics, setRecentMetrics] = useState([]);
  const [evaluations, setEvaluations] = useState([]);

  const loadDashboardData = async () => {
    try {
      const [metricsResp, evalResp] = await Promise.all([
        getRecentMetrics(10).catch(() => ({ metrics: [] })),
        getEvaluations(10).catch(() => ({ evaluations: [] })),
      ]);
      setRecentMetrics(metricsResp.metrics || []);
      setEvaluations(evalResp.evaluations || []);
    } catch {
      // Ignore dashboard refresh issues; the upload flow remains usable.
    }
  };

  useEffect(() => {
    loadDashboardData();
  }, []);

  const handleUpload = async (file) => {
    setUploading(true);
    setProgress(0);
    setError(null);
    setProcessingStatus("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await uploadMeeting(formData);
      const { meeting_id } = response;
      setProgress(5);
      setProcessingStatus("Processing started...");

      let pollInterval = null;

      const poll = async () => {
        try {
          const prog = await getMeetingProgress(meeting_id);
          setProgress(prog.progress ?? 0);
          setProcessingStatus(prog.message || prog.current_stage || "");

          if (prog.status === "completed") {
            if (pollInterval) clearInterval(pollInterval);
            const meetingData = await getMeeting(meeting_id);
            setSelectedMeeting(meetingData);
            setMeetings((prev) => [meetingData, ...prev.filter((m) => m?.meeting?.id !== meeting_id)]);
            setUploading(false);
            setTimeout(() => {
              setProgress(0);
              setProcessingStatus("");
            }, 1000);
            loadDashboardData();
          } else if (prog.status === "failed") {
            if (pollInterval) clearInterval(pollInterval);
            setError(prog.message || "Processing failed");
            setUploading(false);
            setProcessingStatus("");
          }
        } catch (err) {
          if (pollInterval) clearInterval(pollInterval);
          setError(err.response?.data?.detail || "Polling error");
          setUploading(false);
          setProcessingStatus("");
        }
      };

      await poll();
      pollInterval = setInterval(poll, 1500);
    } catch (err) {
      setError(err.response?.data?.detail || "Upload failed");
      setUploading(false);
      setProcessingStatus("");
    }
  };

  const refreshMeeting = async (id) => {
    try {
      const data = await getMeeting(id);
      setSelectedMeeting(data);
    } catch {
      setError("Failed to load meeting");
    }
  };

  const exportData = async (formatName) => {
    const meetingId = selectedMeeting?.meeting?.id;
    if (!meetingId) return;
    try {
      const blob = await exportMeeting(meetingId, formatName);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `meeting-${meetingId}.${formatName}`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch {
      setError("Export failed");
    }
  };

  const metadata = selectedMeeting?.metadata || {};
  const tasks = selectedMeeting?.tasks || [];
  const segments = selectedMeeting?.segments || [];

  const chartData = useMemo(
    () => [
      { name: "Transcription", time: Number(metadata.transcribe_time_sec || 0) },
      { name: "Task extraction", time: Number(metadata.task_time_sec || 0) },
      { name: "Assignment", time: Number(metadata.assign_time_sec || 0) },
    ],
    [metadata]
  );

  const groupedModels = useMemo(() => {
    const map = new Map();
    for (const row of recentMetrics) {
      const key = `${row.model_whisper || "unknown"} / ${row.model_task || "unknown"}`;
      if (!map.has(key)) {
        map.set(key, {
          key,
          model_whisper: row.model_whisper || "unknown",
          model_task: row.model_task || "unknown",
          runs: 0,
          total_latency: 0,
          tasks_count: 0,
          avg_confidence: 0,
        });
      }
      const item = map.get(key);
      item.runs += 1;
      item.total_latency += Number(row.total_latency_sec || 0);
      item.tasks_count += Number(row.tasks_count || 0);
      item.avg_confidence += Number(row.transcript_confidence || 0);
    }
    return [...map.values()].map((x) => ({
      ...x,
      avg_latency: x.runs ? x.total_latency / x.runs : 0,
      avg_tasks: x.runs ? x.tasks_count / x.runs : 0,
      avg_confidence: x.runs ? x.avg_confidence / x.runs : 0,
    }));
  }, [recentMetrics]);

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
        Meeting Secretary
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Upload Audio
            </Typography>
            <Button
              variant="contained"
              component="label"
              startIcon={<Upload />}
              fullWidth
              disabled={uploading}
              sx={{ mb: 2 }}
            >
              {uploading ? "Processing..." : "Select Audio File"}
              <input
                type="file"
                hidden
                accept="audio/*"
                onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
              />
            </Button>

            {uploading && (
              <>
                <LinearProgress variant="determinate" value={progress} sx={{ mb: 2 }} />
                <Typography variant="body2" color="text.secondary">
                  {processingStatus}
                </Typography>
              </>
            )}

            <Divider sx={{ my: 2 }} />
            <Typography variant="subtitle2" color="text.secondary">
              Supported formats: MP3, WAV, M4A, FLAC
            </Typography>
          </Paper>

          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Meetings
            </Typography>
            <List dense>
              {meetings.slice(0, 10).map((m) => (
                <ListItem
                  button
                  key={m.meeting?.id}
                  onClick={() => setSelectedMeeting(m)}
                  selected={selectedMeeting?.meeting?.id === m.meeting?.id}
                >
                  <ListItemText
                    primary={`Meeting #${m.meeting?.id}`}
                    secondary={format(new Date(m.meeting?.created_at || Date.now()), "PPpp")}
                  />
                </ListItem>
              ))}
              {meetings.length === 0 && (
                <Typography variant="body2" color="text.secondary">
                  No meetings yet
                </Typography>
              )}
            </List>
          </Paper>

          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Evaluation Runs
            </Typography>
            <List dense>
              {evaluations.slice(0, 6).map((run) => (
                <ListItem key={run.id}>
                  <ListItemText
                    primary={`Run #${run.id} — ${safeNumber(run.overall_score, 1)} / 100`}
                    secondary={`${run.model_whisper} • ${run.model_task}`}
                  />
                </ListItem>
              ))}
              {evaluations.length === 0 && (
                <Typography variant="body2" color="text.secondary">
                  No evaluation runs available yet
                </Typography>
              )}
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={8}>
          {selectedMeeting ? (
            <Box>
              <Paper sx={{ mb: 3 }}>
                <Box sx={{ borderBottom: 1, borderColor: "divider", px: 3, py: 1 }}>
                  <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)}>
                    <Tab label="Transcript" value="transcript" icon={<Timeline />} />
                    <Tab label="Tasks" value="tasks" icon={<Assessment />} />
                    <Tab label="Metrics" value="metrics" icon={<BarChartIcon />} />
                  </Tabs>
                </Box>

                <Box sx={{ p: 3 }}>
                  {activeTab === "transcript" && (
                    <TranscriptView segments={segments} transcript={selectedMeeting.transcript} metadata={metadata} />
                  )}

                  {activeTab === "tasks" && (
                    <TasksView tasks={tasks} />
                  )}

                  {activeTab === "metrics" && (
                    <MetricsView
                      metadata={metadata}
                      recentMetrics={recentMetrics}
                      groupedModels={groupedModels}
                      evaluations={evaluations}
                    />
                  )}
                </Box>
              </Paper>

              <Box sx={{ display: "flex", gap: 2, mb: 3 }}>
                <Button variant="outlined" startIcon={<Download />} onClick={() => exportData("json")}>
                  JSON
                </Button>
                <Button variant="outlined" startIcon={<Download />} onClick={() => exportData("csv")}>
                  CSV
                </Button>
                <Button variant="outlined" startIcon={<Download />} onClick={() => exportData("md")}>
                  Markdown
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<Refresh />}
                  onClick={() => refreshMeeting(selectedMeeting.meeting.id)}
                >
                  Refresh
                </Button>
              </Box>
            </Box>
          ) : (
            <Paper sx={{ p: 6, textAlign: "center" }}>
              <Typography variant="h6" color="text.secondary">
                Upload an audio file to get started
              </Typography>
            </Paper>
          )}
        </Grid>
      </Grid>

      <Snackbar open={!!error} autoHideDuration={7000} onClose={() => setError(null)}>
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      </Snackbar>
    </Container>
  );
}

function TranscriptView({ segments, transcript, metadata }) {
  return (
    <Box>
      <Typography variant="subtitle2" gutterBottom>
        {metadata?.language ? `Language: ${metadata.language}` : "Language: unknown"}
        {metadata?.has_diarization ? " • Diarization enabled" : ""}
      </Typography>

      <Paper variant="outlined" sx={{ p: 2, mb: 2, whiteSpace: "pre-wrap" }}>
        {transcript || "No transcript available"}
      </Paper>

      {segments?.length ? (
        <Box>
          <Typography variant="h6" gutterBottom>
            Segments
          </Typography>
          {segments.map((segment, idx) => (
            <Paper key={idx} variant="outlined" sx={{ p: 2, mb: 1 }}>
              <Typography variant="caption" color="text.secondary">
                {safeNumber(segment.start, 1)}s — {safeNumber(segment.end, 1)}s
                {segment.speaker ? ` • ${segment.speaker}` : ""}
                {typeof segment.confidence === "number" ? ` • conf ${safeNumber(segment.confidence, 2)}` : ""}
              </Typography>
              <Typography>{segment.text}</Typography>
            </Paper>
          ))}
        </Box>
      ) : null}
    </Box>
  );
}

function TasksView({ tasks }) {
  if (!tasks?.length) {
    return <Typography color="text.secondary">No tasks extracted yet.</Typography>;
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Extracted Tasks
      </Typography>
      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell width="5%">#</TableCell>
              <TableCell width="55%">Task</TableCell>
              <TableCell width="20%">Assignee</TableCell>
              <TableCell width="20%">Deadline</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {tasks.map((task, index) => (
              <TableRow key={task.id || index}>
                <TableCell>{index + 1}</TableCell>
                <TableCell>{taskLabel(task)}</TableCell>
                <TableCell>{task.assignee || task.assignee_hint || "—"}</TableCell>
                <TableCell>{task.deadline || task.deadline_hint || "—"}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

function MetricsCard({ title, value, caption }) {
  return (
    <Card>
      <CardContent>
        <Typography variant="overline" color="text.secondary">
          {title}
        </Typography>
        <Typography variant="h5" sx={{ fontWeight: 700 }}>
          {value}
        </Typography>
        {caption ? (
          <Typography variant="body2" color="text.secondary">
            {caption}
          </Typography>
        ) : null}
      </CardContent>
    </Card>
  );
}

function MetricsView({ metadata, recentMetrics, groupedModels, evaluations }) {
  const modelLabel = `${metadata?.model_whisper || "unknown"} / ${metadata?.model_task || "unknown"}`;

  const recentRunsChart = groupedModels.map((x) => ({
    name: x.key,
    avgLatency: Number(x.avg_latency || 0),
    avgTasks: Number(x.avg_tasks || 0),
  }));

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <MetricsCard
          title="Current model pair"
          value={modelLabel}
          caption="Used for this meeting"
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <MetricsCard
          title="Transcript confidence"
          value={typeof metadata?.transcript_confidence === "number" ? safeNumber(metadata.transcript_confidence, 2) : "—"}
          caption="Heuristic Whisper confidence"
        />
      </Grid>

      <Grid item xs={12} md={3}>
        <MetricsCard title="Audio size" value={metadata?.audio_size_bytes ? `${(metadata.audio_size_bytes / 1024 / 1024).toFixed(2)} MB` : "—"} />
      </Grid>
      <Grid item xs={12} md={3}>
        <MetricsCard title="Duration" value={metadata?.audio_duration_sec ? `${safeNumber(metadata.audio_duration_sec, 1)} s` : "—"} />
      </Grid>
      <Grid item xs={12} md={3}>
        <MetricsCard title="Segments" value={metadata?.segments_count ?? "—"} />
      </Grid>
      <Grid item xs={12} md={3}>
        <MetricsCard title="Tasks" value={metadata?.tasks_count ?? "—"} />
      </Grid>

      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Processing time breakdown
            </Typography>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={chartDataFromMetadata(metadata)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="time" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Recent processing runs
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Model</TableCell>
                    <TableCell align="right">Latency</TableCell>
                    <TableCell align="right">Tasks</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {groupedModels.slice(0, 6).map((row) => (
                    <TableRow key={row.key}>
                      <TableCell>{row.key}</TableCell>
                      <TableCell align="right">{safeNumber(row.avg_latency, 1)} s</TableCell>
                      <TableCell align="right">{safeNumber(row.avg_tasks, 1)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Evaluation runs
            </Typography>
            {evaluations.length ? (
              <List dense>
                {evaluations.slice(0, 6).map((run) => (
                  <ListItem key={run.id}>
                    <ListItemText
                      primary={`Run #${run.id} — score ${safeNumber(run.overall_score, 1)}/100`}
                      secondary={`${run.model_whisper} • ${run.model_task}`}
                    />
                  </ListItem>
                ))}
              </List>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No evaluation runs yet. After generating gold annotations, this tab can compare models by WER, task F1, and the weighted overall score.
              </Typography>
            )}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
}

function chartDataFromMetadata(metadata) {
  return [
    { name: "Transcription", time: Number(metadata?.transcribe_time_sec || 0) },
    { name: "Task extract", time: Number(metadata?.task_time_sec || 0) },
    { name: "Assignment", time: Number(metadata?.assign_time_sec || 0) },
  ];
}
