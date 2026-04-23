import React, { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  Grid,
  LinearProgress,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Paper,
  Snackbar,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import {
  Upload,
  Download,
  Refresh,
  Timeline,
  Assessment,
  Analytics,
  BarChart as BarChartIcon,
  Storage,
  Speed,
  EventNote,
  RecordVoiceOver,
} from "@mui/icons-material";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { format } from "date-fns";

import {
  uploadMeeting,
  getMeeting,
  getMeetingProgress,
  exportMeeting,
  getRecentMetrics,
  getEvaluations,
  getMeetings,
  evaluateMeeting,
} from "./api";

function safeNumber(value, digits = 1) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "—";
  return num.toFixed(digits);
}

function taskLabel(task) {
  return task?.description || "Untitled task";
}

function getMeetingId(entry) {
  return entry?.meeting?.id ?? entry?.meeting_id ?? entry?.id ?? null;
}

function asDate(value) {
  if (!value) return null;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

function formatDate(value) {
  const d = asDate(value);
  if (!d) return "—";
  return format(d, "yyyy-MM-dd HH:mm");
}

function fmtPct(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return `${(n * 100).toFixed(1)}%`;
}

const METRIC_PALETTE = ["#1976d2", "#9c27b0", "#2e7d32", "#ed6c02", "#d32f2f", "#0288d1"];

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
  const [evaluating, setEvaluating] = useState(false);
  const [evaluationMessage, setEvaluationMessage] = useState("");

  const loadDashboardData = async (preferredMeetingId = null) => {
    try {
      const [metricsResp, evalResp, meetingsResp] = await Promise.all([
        getRecentMetrics(20).catch(() => ({ metrics: [] })),
        getEvaluations(20).catch(() => ({ evaluations: [] })),
        getMeetings(20).catch(() => ({ meetings: [] })),
      ]);

      const meetingsList = meetingsResp.meetings || [];
      setRecentMetrics(metricsResp.metrics || []);
      setEvaluations(evalResp.evaluations || []);
      setMeetings(meetingsList);

      if (preferredMeetingId) {
        const current = meetingsList.find((m) => getMeetingId(m) === preferredMeetingId);
        if (current) {
          setSelectedMeeting(current);
          return;
        }
      }

      if (!selectedMeeting && meetingsList.length > 0) {
        setSelectedMeeting(meetingsList[0]);
        setActiveTab("transcript");
      }
    } catch {
      // Keep UI usable even if dashboard refresh fails.
    }
  };

  useEffect(() => {
    loadDashboardData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleUpload = async (file) => {
    setUploading(true);
    setProgress(0);
    setError(null);
    setProcessingStatus("Uploading file...");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await uploadMeeting(formData, (uploadPercent) => {
        setProgress(Math.round((uploadPercent / 100) * 5));
      });

      const { meeting_id } = response;
      setProgress(5);
      setProcessingStatus("Processing started...");

      let pollInterval = null;

      const poll = async () => {
        try {
          const prog = await getMeetingProgress(meeting_id);
          setProgress(5 + Math.round((prog.progress ?? 0) * 0.95));
          setProcessingStatus(prog.message || prog.current_stage || "");

          if (prog.status === "completed") {
            if (pollInterval) clearInterval(pollInterval);
            const meetingData = await getMeeting(meeting_id);
            setSelectedMeeting(meetingData);
            setMeetings((prev) => [meetingData, ...prev.filter((m) => getMeetingId(m) !== meeting_id)]);
            setUploading(false);
            setTimeout(() => {
              setProgress(0);
              setProcessingStatus("");
            }, 1000);
            loadDashboardData(meeting_id);
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

      pollInterval = setInterval(poll, 1500);
      await poll();
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
      setMeetings((prev) => [data, ...prev.filter((m) => getMeetingId(m) !== id)]);
    } catch {
      setError("Failed to load meeting");
    }
  };

  const exportData = async (formatName) => {
    const meetingId = getMeetingId(selectedMeeting);
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

  const runEvaluation = async () => {
    const meetingId = getMeetingId(selectedMeeting);
    if (!meetingId) return;
    setEvaluating(true);
    setEvaluationMessage("Running evaluation...");
    try {
      const res = await evaluateMeeting(meetingId);
      setEvaluationMessage(`Evaluation completed: ${safeNumber(res.overall_quality_score, 1)}/100`);
      await loadDashboardData(meetingId);
    } catch (err) {
      setError(err.response?.data?.detail || "Evaluation failed");
    } finally {
      setEvaluating(false);
      setTimeout(() => setEvaluationMessage(""), 2500);
    }
  };

  const selectedMeetingId = getMeetingId(selectedMeeting);
  const selectedMetadata = selectedMeeting?.metadata || {};
  const selectedSegments = selectedMeeting?.segments || [];
  const selectedTasks = selectedMeeting?.tasks || [];
  const transcript = selectedMeeting?.meeting?.transcript || selectedMeeting?.transcript || "";
  const speakerAliases = selectedMetadata?.speaker_aliases || {};

  const selectedMeetingEvaluations = useMemo(() => {
    if (!selectedMeetingId) return [];
    return evaluations.filter((run) => run.meeting_id === selectedMeetingId);
  }, [evaluations, selectedMeetingId]);

  const latestEvaluation = selectedMeetingEvaluations[0] || null;

  const processingRows = useMemo(() => {
    return [...recentMetrics]
      .sort((a, b) => Number(a.meeting_id || 0) - Number(b.meeting_id || 0))
      .map((m) => ({
        meeting_id: m.meeting_id,
        total: Number(m.total_latency_sec || 0),
        transcribe: Number(m.transcribe_latency_sec || 0),
        task: Number(m.task_latency_sec || 0),
        assign: Number(m.assign_latency_sec || 0),
        confidence: Number(m.transcript_confidence || 0),
        tasks: Number(m.tasks_count || 0),
      }));
  }, [recentMetrics]);

  const evaluationRows = useMemo(() => {
    return [...evaluations].map((e) => ({
      run: `#${e.id}`,
      overall: Number(e.overall_score || 0),
      wer: Number(e.wer ?? 0),
      cer: Number(e.cer ?? 0),
      f1: Number(e.task_set_f1 ?? 0),
      precision: Number(e.task_set_precision ?? 0),
      recall: Number(e.task_set_recall ?? 0),
      fallback: e.task_fallback_used ? 1 : 0,
      provider: e.task_provider || "unknown",
      parse: e.task_parse_stage || "—",
      model: e.model_task || "unknown",
    }));
  }, [evaluations]);

  const taskComparison = latestEvaluation
    ? [
        { name: "Matched", value: Number(latestEvaluation.matched_tasks || 0) },
        { name: "Predicted", value: Number(latestEvaluation.predicted_tasks || 0) },
        { name: "Gold", value: Number(latestEvaluation.gold_tasks || 0) },
      ]
    : [];

  const scoreTrend = evaluationRows.map((r) => ({
    name: r.run,
    overall: r.overall,
    wer: r.wer,
    f1: r.f1,
  }));

  return (
    <Container maxWidth="xl" sx={{ py: 3 }}>
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 2, mb: 3 }}>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 800 }}>
            Meeting Secretary
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Automated transcription, task extraction and evaluation dashboard
          </Typography>
        </Box>
        <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
          <Button component="label" variant="contained" startIcon={<Upload />} disabled={uploading}>
            Upload audio
            <input
              hidden
              type="file"
              accept="audio/*"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleUpload(file);
                e.target.value = "";
              }}
            />
          </Button>
          <Button variant="outlined" startIcon={<Refresh />} onClick={() => loadDashboardData(selectedMeetingId)}>
            Refresh dashboard
          </Button>
        </Box>
      </Box>

      {uploading ? (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            {processingStatus || "Processing..."}
          </Typography>
          <LinearProgress variant="determinate" value={progress} />
          <Typography variant="caption" color="text.secondary">
            {progress}%
          </Typography>
        </Paper>
      ) : null}

      {evaluationMessage ? (
        <Alert severity="success" sx={{ mb: 3 }}>
          {evaluationMessage}
        </Alert>
      ) : null}

      <Grid container spacing={3}>
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2, mb: 2 }} variant="outlined">
            <Typography variant="h6" gutterBottom>
              Meetings
            </Typography>
            <List dense sx={{ maxHeight: 680, overflow: "auto" }}>
              {meetings.map((meeting) => {
                const id = getMeetingId(meeting);
                const status = meeting?.metadata?.status || meeting?.metadata?.current_stage || "processed";
                const selected = id === selectedMeetingId;
                return (
                  <ListItem key={id} disablePadding sx={{ mb: 0.5 }}>
                    <ListItemButton
                      selected={selected}
                      onClick={() => {
                        setSelectedMeeting(meeting);
                        setActiveTab("transcript");
                      }}
                    >
                      <ListItemText
                        primary={`Meeting #${id}`}
                        secondary={
                          <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
                            <Typography variant="caption" color="text.secondary">
                              {formatDate(meeting?.meeting?.created_at || meeting?.metadata?.created_at)}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {status}
                            </Typography>
                          </Box>
                        }
                      />
                    </ListItemButton>
                  </ListItem>
                );
              })}
              {!meetings.length ? (
                <Typography variant="body2" color="text.secondary" sx={{ px: 2, py: 1 }}>
                  No meetings yet.
                </Typography>
              ) : null}
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={9}>
          {selectedMeeting ? (
            <Paper sx={{ mb: 3 }} variant="outlined">
              <Box sx={{ borderBottom: 1, borderColor: "divider", px: 3, py: 1 }}>
                <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} variant="scrollable" scrollButtons="auto">
                  <Tab label="Transcript" value="transcript" icon={<Timeline />} iconPosition="start" />
                  <Tab label="Tasks" value="tasks" icon={<Assessment />} iconPosition="start" />
                  <Tab label="Metrics" value="metrics" icon={<BarChartIcon />} iconPosition="start" />
                  <Tab label="Analytics" value="analytics" icon={<Analytics />} iconPosition="start" />
                </Tabs>
              </Box>

              <Box sx={{ p: 3 }}>
                {activeTab === "transcript" && (
                  <TranscriptView segments={selectedSegments} transcript={transcript} metadata={selectedMetadata} speakerAliases={speakerAliases} />
                )}

                {activeTab === "tasks" && <TasksView tasks={selectedTasks} />}

                {activeTab === "metrics" && (
                  <MetricsView
                    metadata={selectedMetadata}
                    latestEvaluation={latestEvaluation}
                    runEvaluation={runEvaluation}
                    evaluating={evaluating}
                  />
                )}

                {activeTab === "analytics" && (
                  <AnalyticsView
                    processingRows={processingRows}
                    evaluationRows={evaluationRows}
                    taskComparison={taskComparison}
                    latestEvaluation={latestEvaluation}
                  />
                )}
              </Box>
            </Paper>
          ) : (
            <Paper sx={{ p: 6, textAlign: "center" }} variant="outlined">
              <Typography variant="h6" color="text.secondary">
                Upload an audio file to get started
              </Typography>
            </Paper>
          )}

          <Box sx={{ display: "flex", gap: 2, mb: 3, flexWrap: "wrap" }}>
            <Button variant="outlined" startIcon={<Download />} onClick={() => exportData("json")} disabled={!selectedMeeting}>
              JSON
            </Button>
            <Button variant="outlined" startIcon={<Download />} onClick={() => exportData("csv")} disabled={!selectedMeeting}>
              CSV
            </Button>
            <Button variant="outlined" startIcon={<Download />} onClick={() => exportData("md")} disabled={!selectedMeeting}>
              Markdown
            </Button>
            <Button variant="outlined" startIcon={<Refresh />} onClick={() => refreshMeeting(selectedMeetingId)} disabled={!selectedMeeting}>
              Refresh meeting
            </Button>
            <Button variant="outlined" startIcon={<EventNote />} onClick={runEvaluation} disabled={!selectedMeeting || evaluating}>
              Run evaluation
            </Button>
          </Box>
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

function TranscriptView({ segments, transcript, metadata, speakerAliases }) {
  return (
    <Box>
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} md={3}>
          <Card variant="outlined"><CardContent><Typography variant="overline" color="text.secondary">Language</Typography><Typography variant="h6">{metadata?.language || "unknown"}</Typography></CardContent></Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card variant="outlined"><CardContent><Typography variant="overline" color="text.secondary">Diarization</Typography><Typography variant="h6">{metadata?.has_diarization ? "enabled" : "off"}</Typography></CardContent></Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card variant="outlined"><CardContent><Typography variant="overline" color="text.secondary">Transcript confidence</Typography><Typography variant="h6">{typeof metadata?.transcript_confidence === "number" ? safeNumber(metadata.transcript_confidence, 2) : "—"}</Typography></CardContent></Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card variant="outlined"><CardContent><Typography variant="overline" color="text.secondary">Speaker aliases</Typography><Typography variant="h6">{Object.keys(speakerAliases || {}).length}</Typography></CardContent></Card>
        </Grid>
      </Grid>

      {Object.keys(speakerAliases || {}).length ? (
        <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Inferred speaker aliases
          </Typography>
          <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
            {Object.entries(speakerAliases).map(([speaker, name]) => (
              <Chip key={speaker} label={`${speaker} → ${name}`} size="small" />
            ))}
          </Box>
        </Paper>
      ) : null}

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
              <TableCell width="4%">#</TableCell>
              <TableCell width="42%">Task</TableCell>
              <TableCell width="16%">Assignee</TableCell>
              <TableCell width="14%">Deadline</TableCell>
              <TableCell width="12%">Source</TableCell>
              <TableCell width="12%">Speaker</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {tasks.map((task, index) => (
              <TableRow key={task.id || index}>
                <TableCell>{index + 1}</TableCell>
                <TableCell>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>{taskLabel(task)}</Typography>
                  {task.source_snippet ? <Typography variant="caption" color="text.secondary">{task.source_snippet}</Typography> : null}
                </TableCell>
                <TableCell>{task.assignee || task.assignee_hint || "—"}</TableCell>
                <TableCell>{task.deadline || task.deadline_hint || "—"}</TableCell>
                <TableCell>{task.source || "—"}</TableCell>
                <TableCell>{task.speaker_hint || task.speaker_resolved || "—"}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

function MetricsCard({ title, value, caption, icon }) {
  return (
    <Card variant="outlined">
      <CardContent>
        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 2 }}>
          <Box>
            <Typography variant="overline" color="text.secondary">
              {title}
            </Typography>
            <Typography variant="h5" sx={{ fontWeight: 800 }}>
              {value}
            </Typography>
            {caption ? <Typography variant="body2" color="text.secondary">{caption}</Typography> : null}
          </Box>
          {icon ? <Box sx={{ color: "primary.main" }}>{icon}</Box> : null}
        </Box>
      </CardContent>
    </Card>
  );
}

function MetricsView({ metadata, latestEvaluation, runEvaluation, evaluating }) {
  const modelLabel = `${metadata?.model_whisper || "unknown"} / ${metadata?.model_task || "unknown"}`;
  const taskProvider = metadata?.task_provider || "unknown";
  const parseStage = metadata?.task_parse_stage || "—";

  const processBreakdown = [
    { name: "Transcription", value: Number(metadata?.transcribe_time_sec || 0) },
    { name: "Task extraction", value: Number(metadata?.task_time_sec || 0) },
    { name: "Assignment", value: Number(metadata?.assign_time_sec || 0) },
  ];

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} md={4}>
        <MetricsCard title="Model pair" value={modelLabel} caption="Current meeting" icon={<Storage />} />
      </Grid>
      <Grid item xs={12} md={4}>
        <MetricsCard title="Transcript confidence" value={typeof metadata?.transcript_confidence === "number" ? safeNumber(metadata.transcript_confidence, 2) : "—"} caption="Heuristic Whisper confidence" icon={<RecordVoiceOver />} />
      </Grid>
      <Grid item xs={12} md={4}>
        <MetricsCard title="Task provider" value={taskProvider} caption={`parse: ${parseStage}`} icon={<Assessment />} />
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

      <Grid item xs={12} md={6}>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Processing time breakdown
            </Typography>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={processBreakdown}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#1976d2" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Latest evaluation
            </Typography>
            {latestEvaluation ? (
              <Box sx={{ display: "grid", gap: 1.5 }}>
                <Typography variant="body2">Overall score: <strong>{safeNumber(latestEvaluation.overall_score, 1)}</strong>/100</Typography>
                <Typography variant="body2">WER: <strong>{safeNumber(latestEvaluation.wer, 3)}</strong> • CER: <strong>{safeNumber(latestEvaluation.cer, 3)}</strong></Typography>
                <Typography variant="body2">Task F1: <strong>{safeNumber(latestEvaluation.task_set_f1, 3)}</strong> • Precision: <strong>{safeNumber(latestEvaluation.task_set_precision, 3)}</strong> • Recall: <strong>{safeNumber(latestEvaluation.task_set_recall, 3)}</strong></Typography>
                <Typography variant="body2">Matched tasks: <strong>{latestEvaluation.matched_tasks ?? 0}</strong> / {latestEvaluation.gold_tasks ?? 0}</Typography>
                <Typography variant="body2">Provider: <strong>{latestEvaluation.task_provider || "unknown"}</strong> • parse: <strong>{latestEvaluation.task_parse_stage || "—"}</strong> • fallback: <strong>{String(Boolean(latestEvaluation.task_fallback_used))}</strong></Typography>
                <Button variant="contained" onClick={runEvaluation} disabled={evaluating} startIcon={<Analytics />}>
                  {evaluating ? "Running..." : "Re-run evaluation"}
                </Button>
              </Box>
            ) : (
              <Box sx={{ display: "grid", gap: 1.5 }}>
                <Typography variant="body2" color="text.secondary">
                  No evaluation run for this meeting yet.
                </Typography>
                <Button variant="contained" onClick={runEvaluation} disabled={evaluating} startIcon={<Analytics />}>
                  {evaluating ? "Running..." : "Run evaluation"}
                </Button>
              </Box>
            )}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
}

function AnalyticsView({ processingRows, evaluationRows, taskComparison }) {
  const topEvaluations = evaluationRows.slice(0, 12).reverse();

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Latency by meeting
            </Typography>
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={processingRows}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="meeting_id" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area type="monotone" dataKey="transcribe" stackId="1" fill="#1976d2" stroke="#1976d2" name="Transcribe" />
                <Area type="monotone" dataKey="task" stackId="1" fill="#9c27b0" stroke="#9c27b0" name="Task" />
                <Area type="monotone" dataKey="assign" stackId="1" fill="#2e7d32" stroke="#2e7d32" name="Assign" />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Evaluation score trend
            </Typography>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={topEvaluations}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="run" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="overall" stroke="#1976d2" name="Overall" strokeWidth={2} />
                <Line type="monotone" dataKey="f1" stroke="#2e7d32" name="Task F1" strokeWidth={2} />
                <Line type="monotone" dataKey="wer" stroke="#d32f2f" name="WER" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Predicted vs gold tasks
            </Typography>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={taskComparison}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value">
                  {taskComparison.map((entry, idx) => (
                    <Cell key={entry.name} fill={METRIC_PALETTE[idx % METRIC_PALETTE.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Evaluation runs summary
            </Typography>
            {evaluationRows.length ? (
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Run</TableCell>
                      <TableCell>Provider</TableCell>
                      <TableCell>Parse</TableCell>
                      <TableCell align="right">Score</TableCell>
                      <TableCell align="right">F1</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {evaluationRows.slice(0, 8).map((row) => (
                      <TableRow key={row.run}>
                        <TableCell>{row.run}</TableCell>
                        <TableCell>{row.provider}</TableCell>
                        <TableCell>{row.parse}</TableCell>
                        <TableCell align="right">{safeNumber(row.overall, 1)}</TableCell>
                        <TableCell align="right">{safeNumber(row.f1, 3)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No evaluation data yet.
              </Typography>
            )}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
}
