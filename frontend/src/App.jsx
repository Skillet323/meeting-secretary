import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  Grid,
  IconButton,
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
  Tooltip as MuiTooltip,
} from "@mui/material";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import {
  Upload,
  Download,
  Refresh,
  Timeline,
  Assessment,
  Analytics,
  BarChart as BarChartIcon,
  Storage,
  EventNote,
  RecordVoiceOver,
  Dashboard as DashboardIcon,
  Circle,
  TrendingUp,
  Task,
  AccessTime,
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
  ReferenceLine,
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
  getStats,
} from "./api";

// ─── dark theme ───────────────────────────────────────────────────────────────
const theme = createTheme({
  palette: {
    mode: "dark",
    primary: { main: "#4fc3f7" },
    secondary: { main: "#a78bfa" },
    background: { default: "#0d1117", paper: "#161b22" },
    text: { primary: "#e6edf3", secondary: "#8b949e" },
    divider: "#30363d",
  },
  shape: { borderRadius: 8 },
  components: {
    MuiPaper: { styleOverrides: { root: { backgroundImage: "none", border: "1px solid #30363d" } } },
    MuiCard: { styleOverrides: { root: { backgroundImage: "none", border: "1px solid #30363d" } } },
    MuiTableCell: { styleOverrides: { root: { borderColor: "#30363d" } } },
  },
});

// ─── constants ────────────────────────────────────────────────────────────────
const PALETTE = ["#4fc3f7", "#a78bfa", "#34d399", "#f59e0b", "#f87171", "#38bdf8"];
const GREEN = "#34d399";
const RED = "#f87171";
const YELLOW = "#f59e0b";

// ─── helpers ──────────────────────────────────────────────────────────────────
function safeN(v, d = 1) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(d) : "—";
}
function getMeetingId(e) {
  return e?.meeting?.id ?? e?.meeting_id ?? e?.id ?? null;
}
function asDate(v) {
  if (!v) return null;
  const d = new Date(v);
  return isNaN(d) ? null : d;
}
function formatDate(v) {
  const d = asDate(v);
  return d ? format(d, "yyyy-MM-dd HH:mm") : "—";
}

// ─── chart tooltip ────────────────────────────────────────────────────────────
function CT({ active, payload, label, unit = "" }) {
  if (!active || !payload?.length) return null;
  return (
    <Paper sx={{ p: 1.5 }}>
      <Typography variant="caption" color="text.secondary">{label}</Typography>
      {payload.map((p) => (
        <Typography key={p.dataKey} variant="body2" sx={{ color: p.color }}>
          {p.name}: <strong>{safeN(p.value, 2)}{unit}</strong>
        </Typography>
      ))}
    </Paper>
  );
}

// ─── KPI card ─────────────────────────────────────────────────────────────────
function KpiCard({ title, value, sub, icon, color = "#4fc3f7" }) {
  return (
    <Card sx={{ height: "100%", overflow: "hidden", position: "relative" }}>
      <Box sx={{ position: "absolute", left: 0, top: 0, bottom: 0, width: 4, bgcolor: color, borderRadius: "8px 0 0 8px" }} />
      <CardContent sx={{ pl: 3 }}>
        <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <Box>
            <Typography variant="overline" sx={{ color: "text.secondary", fontSize: "0.65rem", letterSpacing: 1 }}>{title}</Typography>
            <Typography variant="h4" sx={{ fontWeight: 800, color, lineHeight: 1.1 }}>{value}</Typography>
            {sub && <Typography variant="caption" color="text.secondary">{sub}</Typography>}
          </Box>
          {icon && <Box sx={{ color, opacity: 0.5, mt: 0.5 }}>{React.cloneElement(icon, { sx: { fontSize: 38 } })}</Box>}
        </Box>
      </CardContent>
    </Card>
  );
}

// ─── status dot ───────────────────────────────────────────────────────────────
function StatusDot({ status }) {
  const color = status === "completed" ? GREEN : status === "processing" ? YELLOW : status === "failed" ? RED : "#8b949e";
  return (
    <Box sx={{ display: "inline-flex", alignItems: "center", gap: 0.5 }}>
      <Circle sx={{ fontSize: 8, color }} />
      <Typography variant="caption" sx={{ color }}>{status || "unknown"}</Typography>
    </Box>
  );
}

// ─── panel ────────────────────────────────────────────────────────────────────
function Panel({ title, children, action, minH = 0 }) {
  return (
    <Card sx={{ height: "100%" }}>
      <CardContent sx={{ display: "flex", flexDirection: "column", height: "100%" }}>
        <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
          <Typography variant="overline" sx={{ color: "text.secondary", fontSize: "0.65rem", letterSpacing: 1, fontWeight: 700 }}>{title}</Typography>
          {action}
        </Box>
        <Box sx={{ flex: 1, minHeight: minH }}>{children}</Box>
      </CardContent>
    </Card>
  );
}

function Empty({ label = "No data yet" }) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", minHeight: 120 }}>
      <Typography variant="body2" color="text.secondary">{label}</Typography>
    </Box>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// DASHBOARD PAGE
// ─────────────────────────────────────────────────────────────────────────────
function DashboardPage({ metrics, evaluations, meetings, stats, onRefresh, refreshing }) {
  const rows = useMemo(
    () =>
      [...metrics]
        .sort((a, b) => Number(a.meeting_id) - Number(b.meeting_id))
        .map((m) => ({
          name: `#${m.meeting_id}`,
          total: +(m.total_latency_sec || 0),
          transcribe: +(m.transcribe_latency_sec || 0),
          task: +(m.task_latency_sec || 0),
          assign: +(m.assign_latency_sec || 0),
          confidence: +(m.transcript_confidence || 0),
          tasks: +(m.tasks_count || 0),
        })),
    [metrics]
  );

  const evalRows = useMemo(
    () =>
      [...evaluations]
        .slice(0, 12)
        .reverse()
        .map((e) => ({
          name: `#${e.id}`,
          overall: +(e.overall_score || 0),
          wer: +(e.wer ?? 0),
          f1: +(e.task_set_f1 ?? 0),
        })),
    [evaluations]
  );

  const avgBreakdown = [
    { name: "Transcription", value: metrics.length ? metrics.reduce((s, m) => s + (m.transcribe_latency_sec || 0), 0) / metrics.length : 0 },
    { name: "Task extract", value: metrics.length ? metrics.reduce((s, m) => s + (m.task_latency_sec || 0), 0) / metrics.length : 0 },
    { name: "Assignment", value: metrics.length ? metrics.reduce((s, m) => s + (m.assign_latency_sec || 0), 0) / metrics.length : 0 },
  ];

  const confRows = rows.filter((r) => r.confidence > 0);
  const taskRows = rows.filter((r) => r.tasks > 0);
  const langData = stats?.language_distribution || [];

  return (
    <Box>
      {/* header */}
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 3 }}>
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 800 }}>System Dashboard</Typography>
          <Typography variant="caption" color="text.secondary">
            Live overview · {format(new Date(), "HH:mm:ss")}
          </Typography>
        </Box>
        <Button size="small" variant="outlined" startIcon={<Refresh />} onClick={onRefresh} disabled={refreshing}>
          Refresh
        </Button>
      </Box>

      {/* KPIs */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={6} md={3}>
          <KpiCard title="Total meetings" value={stats?.total_meetings ?? meetings.length} icon={<EventNote />} color="#4fc3f7" />
        </Grid>
        <Grid item xs={6} md={3}>
          <KpiCard title="Tasks extracted" value={stats?.total_tasks ?? "—"} sub={`avg ${safeN(stats?.avg_tasks_per_meeting, 1)}/meeting`} icon={<Task />} color="#a78bfa" />
        </Grid>
        <Grid item xs={6} md={3}>
          <KpiCard title="Avg latency" value={`${safeN(stats?.avg_total_latency_sec, 1)}s`} icon={<AccessTime />} color="#f59e0b" />
        </Grid>
        <Grid item xs={6} md={3}>
          <KpiCard title="Avg confidence" value={`${safeN((stats?.avg_transcript_confidence || 0) * 100, 1)}%`} icon={<RecordVoiceOver />} color="#34d399" />
        </Grid>
      </Grid>

      {/* Row 1: latency stacked area + donut */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} md={8}>
          <Panel title="Processing latency (s) · per meeting" minH={280}>
            {rows.length ? (
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={rows} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
                  <defs>
                    {["transcribe", "task", "assign"].map((k, i) => (
                      <linearGradient key={k} id={`g-${k}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={PALETTE[i]} stopOpacity={0.4} />
                        <stop offset="95%" stopColor={PALETTE[i]} stopOpacity={0} />
                      </linearGradient>
                    ))}
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                  <XAxis dataKey="name" tick={{ fill: "#8b949e", fontSize: 11 }} />
                  <YAxis tick={{ fill: "#8b949e", fontSize: 11 }} unit="s" />
                  <Tooltip content={<CT unit="s" />} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Area type="monotone" dataKey="transcribe" stackId="1" fill="url(#g-transcribe)" stroke={PALETTE[0]} strokeWidth={2} name="Transcribe" />
                  <Area type="monotone" dataKey="task" stackId="1" fill="url(#g-task)" stroke={PALETTE[1]} strokeWidth={2} name="Task extract" />
                  <Area type="monotone" dataKey="assign" stackId="1" fill="url(#g-assign)" stroke={PALETTE[2]} strokeWidth={2} name="Assign" />
                </AreaChart>
              </ResponsiveContainer>
            ) : <Empty />}
          </Panel>
        </Grid>

        <Grid item xs={12} md={4}>
          <Panel title="Avg latency breakdown" minH={280}>
            {metrics.length ? (
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={avgBreakdown} cx="50%" cy="50%" innerRadius={60} outerRadius={95} paddingAngle={3} dataKey="value">
                    {avgBreakdown.map((_, i) => <Cell key={i} fill={PALETTE[i]} />)}
                  </Pie>
                  <Tooltip formatter={(v) => `${safeN(v, 2)}s`} contentStyle={{ background: "#161b22", border: "1px solid #30363d" }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                </PieChart>
              </ResponsiveContainer>
            ) : <Empty />}
          </Panel>
        </Grid>
      </Grid>

      {/* Row 2: confidence bar + tasks bar */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} md={6}>
          <Panel title="Transcript confidence · per meeting" minH={260}>
            {confRows.length ? (
              <ResponsiveContainer width="100%" height={230}>
                <BarChart data={confRows} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                  <XAxis dataKey="name" tick={{ fill: "#8b949e", fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fill: "#8b949e", fontSize: 11 }} />
                  <Tooltip content={<CT />} />
                  <ReferenceLine y={0.8} stroke={GREEN} strokeDasharray="4 2" label={{ value: "target 0.8", fill: GREEN, fontSize: 10 }} />
                  <Bar dataKey="confidence" name="Confidence" radius={[4, 4, 0, 0]}>
                    {confRows.map((d, i) => <Cell key={i} fill={d.confidence >= 0.8 ? GREEN : d.confidence >= 0.6 ? YELLOW : RED} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <Empty />}
          </Panel>
        </Grid>

        <Grid item xs={12} md={6}>
          <Panel title="Tasks extracted · per meeting" minH={260}>
            {taskRows.length ? (
              <ResponsiveContainer width="100%" height={230}>
                <BarChart data={taskRows} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                  <XAxis dataKey="name" tick={{ fill: "#8b949e", fontSize: 11 }} />
                  <YAxis tick={{ fill: "#8b949e", fontSize: 11 }} allowDecimals={false} />
                  <Tooltip content={<CT />} />
                  <Bar dataKey="tasks" name="Tasks" fill={PALETTE[1]} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <Empty />}
          </Panel>
        </Grid>
      </Grid>

      {/* Row 3: eval trend + language pie */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} md={8}>
          <Panel title="Evaluation score trend · overall / task-F1 / WER" minH={260}>
            {evalRows.length ? (
              <ResponsiveContainer width="100%" height={230}>
                <LineChart data={evalRows} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                  <XAxis dataKey="name" tick={{ fill: "#8b949e", fontSize: 11 }} />
                  <YAxis domain={[0, 100]} tick={{ fill: "#8b949e", fontSize: 11 }} />
                  <Tooltip content={<CT />} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line type="monotone" dataKey="overall" stroke={PALETTE[0]} strokeWidth={2} dot={{ r: 3 }} name="Overall" />
                  <Line type="monotone" dataKey="f1" stroke={GREEN} strokeWidth={2} dot={{ r: 3 }} name="Task F1" />
                  <Line type="monotone" dataKey="wer" stroke={RED} strokeWidth={2} dot={{ r: 3 }} strokeDasharray="4 2" name="WER" />
                </LineChart>
              </ResponsiveContainer>
            ) : <Empty label="No evaluation runs yet" />}
          </Panel>
        </Grid>

        <Grid item xs={12} md={4}>
          <Panel title="Language distribution" minH={260}>
            {langData.length ? (
              <ResponsiveContainer width="100%" height={230}>
                <PieChart>
                  <Pie data={langData} cx="50%" cy="50%" outerRadius={85} paddingAngle={3} dataKey="value" nameKey="name"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`} labelLine={false}>
                    {langData.map((_, i) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                  </Pie>
                  <Tooltip contentStyle={{ background: "#161b22", border: "1px solid #30363d" }} />
                </PieChart>
              </ResponsiveContainer>
            ) : <Empty />}
          </Panel>
        </Grid>
      </Grid>

      {/* Recent meetings table */}
      <Panel title="Recent meetings">
        {meetings.length ? (
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Meeting</TableCell>
                  <TableCell>Date</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell align="right">Tasks</TableCell>
                  <TableCell align="right">Lang</TableCell>
                  <TableCell align="right">Confidence</TableCell>
                  <TableCell align="right">Diarization</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {meetings.slice(0, 12).map((meeting) => {
                  const id = getMeetingId(meeting);
                  const meta = meeting?.metadata || {};
                  return (
                    <TableRow key={id} hover>
                      <TableCell><Typography variant="body2" sx={{ fontWeight: 700 }}>#{id}</Typography></TableCell>
                      <TableCell><Typography variant="caption" color="text.secondary">{formatDate(meeting?.meeting?.created_at || meta?.created_at)}</Typography></TableCell>
                      <TableCell><StatusDot status={meta?.status || "completed"} /></TableCell>
                      <TableCell align="right">{meeting?.tasks?.length ?? meta?.tasks_count ?? "—"}</TableCell>
                      <TableCell align="right">{meta?.language || "—"}</TableCell>
                      <TableCell align="right">
                        {typeof meta?.transcript_confidence === "number"
                          ? `${(meta.transcript_confidence * 100).toFixed(1)}%` : "—"}
                      </TableCell>
                      <TableCell align="right">
                        <Chip label={meta?.has_diarization ? "on" : "off"} size="small"
                          color={meta?.has_diarization ? "success" : "default"} sx={{ fontSize: "0.65rem", height: 20 }} />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        ) : <Empty label="No meetings yet. Upload an audio file to get started." />}
      </Panel>
    </Box>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TRANSCRIPT VIEW
// ─────────────────────────────────────────────────────────────────────────────
function TranscriptView({ segments, transcript, metadata, speakerAliases }) {
  return (
    <Box>
      <Grid container spacing={2} sx={{ mb: 2 }}>
        {[
          { label: "Language", value: metadata?.language || "unknown" },
          { label: "Diarization", value: metadata?.has_diarization ? "enabled" : "off" },
          { label: "Confidence", value: typeof metadata?.transcript_confidence === "number" ? `${(metadata.transcript_confidence * 100).toFixed(1)}%` : "—" },
          { label: "Speaker aliases", value: Object.keys(speakerAliases || {}).length },
        ].map(({ label, value }) => (
          <Grid item xs={6} md={3} key={label}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="overline" color="text.secondary">{label}</Typography>
                <Typography variant="h6">{value}</Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {Object.keys(speakerAliases || {}).length > 0 && (
        <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>Inferred speaker aliases</Typography>
          <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
            {Object.entries(speakerAliases).map(([s, n]) => (
              <Chip key={s} label={`${s} → ${n}`} size="small" color="primary" variant="outlined" />
            ))}
          </Box>
        </Paper>
      )}

      <Paper variant="outlined" sx={{ p: 2, mb: 2, whiteSpace: "pre-wrap", fontFamily: "monospace", fontSize: "0.85rem", maxHeight: 480, overflow: "auto" }}>
        {transcript || "No transcript available"}
      </Paper>

      {segments?.length > 0 && (
        <Box>
          <Typography variant="h6" gutterBottom>Segments ({segments.length})</Typography>
          {segments.map((seg, i) => (
            <Paper key={i} variant="outlined" sx={{ p: 1.5, mb: 0.75 }}>
              <Typography variant="caption" color="text.secondary">
                {safeN(seg.start, 1)}s — {safeN(seg.end, 1)}s
                {seg.speaker ? ` · ${seg.speaker}` : ""}
                {typeof seg.confidence === "number" ? ` · conf ${safeN(seg.confidence, 2)}` : ""}
              </Typography>
              <Typography variant="body2">{seg.text}</Typography>
            </Paper>
          ))}
        </Box>
      )}
    </Box>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TASKS VIEW
// ─────────────────────────────────────────────────────────────────────────────
function TasksView({ tasks }) {
  if (!tasks?.length) return <Typography color="text.secondary">No tasks extracted yet.</Typography>;
  return (
    <Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
        <Typography variant="h6">Extracted Tasks</Typography>
        <Chip label={`${tasks.length} tasks`} color="primary" size="small" />
      </Box>
      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell width="4%">#</TableCell>
              <TableCell width="38%">Task</TableCell>
              <TableCell width="16%">Assignee</TableCell>
              <TableCell width="14%">Deadline</TableCell>
              <TableCell width="16%">Confidence</TableCell>
              <TableCell width="12%">Source</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {tasks.map((task, i) => {
              const raw = (() => { try { return task.raw ? JSON.parse(task.raw) : task; } catch { return task; } })();
              const conf = raw?.assignment_confidence;
              return (
                <TableRow key={task.id || i} hover>
                  <TableCell>{i + 1}</TableCell>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>{task.description || "Untitled"}</Typography>
                    {raw?.source_snippet && (
                      <Typography variant="caption" color="text.secondary" sx={{ fontStyle: "italic" }}>"{raw.source_snippet}"</Typography>
                    )}
                  </TableCell>
                  <TableCell>{task.assignee || raw?.assignee_hint || "—"}</TableCell>
                  <TableCell>{task.deadline || raw?.deadline_hint || "—"}</TableCell>
                  <TableCell>
                    {typeof conf === "number" ? (
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                        <LinearProgress variant="determinate" value={conf * 100} sx={{ flex: 1, height: 6, borderRadius: 3,
                          "& .MuiLinearProgress-bar": { bgcolor: conf > 0.8 ? GREEN : conf > 0.5 ? YELLOW : RED } }} />
                        <Typography variant="caption">{(conf * 100).toFixed(0)}%</Typography>
                      </Box>
                    ) : "—"}
                  </TableCell>
                  <TableCell>
                    <Chip label={raw?.assignee_source || task.source || "—"} size="small" variant="outlined" sx={{ fontSize: "0.6rem", height: 20 }} />
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// METRICS VIEW (per-meeting)
// ─────────────────────────────────────────────────────────────────────────────
function MetricsView({ metadata, latestEvaluation, runEvaluation, evaluating }) {
  const breakdown = [
    { name: "Transcription", value: +(metadata?.transcribe_time_sec || 0) },
    { name: "Task extract", value: +(metadata?.task_time_sec || 0) },
    { name: "Assignment", value: +(metadata?.assign_time_sec || 0) },
  ];
  return (
    <Grid container spacing={3}>
      {[
        { t: "Model pair", v: `${metadata?.model_whisper || "?"} / ${metadata?.model_task || "?"}`, ic: <Storage /> },
        { t: "Confidence", v: typeof metadata?.transcript_confidence === "number" ? `${(metadata.transcript_confidence * 100).toFixed(1)}%` : "—", ic: <RecordVoiceOver /> },
        { t: "Task provider", v: metadata?.task_provider || "—", ic: <Analytics /> },
      ].map(({ t, v, ic }) => (
        <Grid item xs={12} md={4} key={t}>
          <Card variant="outlined">
            <CardContent>
              <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                <Box>
                  <Typography variant="overline" color="text.secondary">{t}</Typography>
                  <Typography variant="h6" sx={{ wordBreak: "break-all" }}>{v}</Typography>
                </Box>
                <Box sx={{ color: "primary.main", opacity: 0.6 }}>{ic}</Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      ))}

      {[
        { t: "Audio size", v: metadata?.audio_size_bytes ? `${(metadata.audio_size_bytes / 1048576).toFixed(2)} MB` : "—" },
        { t: "Duration", v: metadata?.audio_duration_sec ? `${safeN(metadata.audio_duration_sec, 1)}s` : "—" },
        { t: "Segments", v: metadata?.segments_count ?? "—" },
        { t: "Tasks found", v: metadata?.tasks_count ?? "—" },
      ].map(({ t, v }) => (
        <Grid item xs={6} md={3} key={t}>
          <Card variant="outlined"><CardContent><Typography variant="overline" color="text.secondary">{t}</Typography><Typography variant="h6">{v}</Typography></CardContent></Card>
        </Grid>
      ))}

      <Grid item xs={12} md={6}>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>Processing time breakdown</Typography>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={breakdown}>
                <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                <XAxis dataKey="name" tick={{ fill: "#8b949e", fontSize: 12 }} />
                <YAxis unit="s" tick={{ fill: "#8b949e", fontSize: 12 }} />
                <Tooltip formatter={(v) => `${safeN(v, 2)}s`} contentStyle={{ background: "#161b22", border: "1px solid #30363d" }} />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  {breakdown.map((_, i) => <Cell key={i} fill={PALETTE[i]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>Latest evaluation</Typography>
            {latestEvaluation ? (
              <Box sx={{ display: "grid", gap: 1.5 }}>
                <Typography variant="body2">Overall: <strong>{safeN(latestEvaluation.overall_score, 1)}</strong>/100</Typography>
                <Typography variant="body2">WER: <strong>{safeN(latestEvaluation.wer, 3)}</strong> · CER: <strong>{safeN(latestEvaluation.cer, 3)}</strong></Typography>
                <Typography variant="body2">Task F1: <strong>{safeN(latestEvaluation.task_set_f1, 3)}</strong></Typography>
                <Typography variant="body2">Matched: <strong>{latestEvaluation.matched_tasks ?? 0}</strong> / {latestEvaluation.gold_tasks ?? 0}</Typography>
                <Button variant="contained" size="small" startIcon={<Analytics />} onClick={runEvaluation} disabled={evaluating}>
                  {evaluating ? "Running..." : "Re-run evaluation"}
                </Button>
              </Box>
            ) : (
              <Box sx={{ display: "grid", gap: 1.5 }}>
                <Typography variant="body2" color="text.secondary">No evaluation run for this meeting yet.</Typography>
                <Button variant="contained" size="small" startIcon={<Analytics />} onClick={runEvaluation} disabled={evaluating}>
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

// ─────────────────────────────────────────────────────────────────────────────
// MAIN APP
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [meetings, setMeetings] = useState([]);
  const [selectedMeeting, setSelectedMeeting] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadPct, setUploadPct] = useState(0);
  const [processingStatus, setProcessingStatus] = useState("");
  const [recentMetrics, setRecentMetrics] = useState([]);
  const [evaluations, setEvaluations] = useState([]);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [evaluating, setEvaluating] = useState(false);
  const [evalMsg, setEvalMsg] = useState("");
  const [topTab, setTopTab] = useState("meetings");
  const [meetingTab, setMeetingTab] = useState("transcript");
  const [refreshing, setRefreshing] = useState(false);

  const loadAll = useCallback(async (preferredId = null) => {
    setRefreshing(true);
    try {
      const [metricsR, evalR, meetingsR, statsR] = await Promise.all([
        getRecentMetrics(50).catch(() => ({ metrics: [] })),
        getEvaluations(20).catch(() => ({ evaluations: [] })),
        getMeetings(20).catch(() => ({ meetings: [] })),
        getStats().catch(() => null),
      ]);
      const list = meetingsR.meetings || [];
      setRecentMetrics(metricsR.metrics || []);
      setEvaluations(evalR.evaluations || []);
      setMeetings(list);
      if (statsR) setStats(statsR);

      if (preferredId) {
        const cur = list.find((m) => getMeetingId(m) === preferredId);
        if (cur) { setSelectedMeeting(cur); setRefreshing(false); return; }
      }
      setSelectedMeeting((prev) => (!prev && list.length ? list[0] : prev));
    } catch { /* ignore */ } finally { setRefreshing(false); }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => { loadAll(); }, [loadAll]);

  const handleUpload = async (file) => {
    setUploading(true);
    setUploadPct(0);
    setError(null);
    setProcessingStatus("Uploading file...");
    try {
      const fd = new FormData();
      fd.append("file", file);
      const { meeting_id } = await uploadMeeting(fd, (p) => setUploadPct(Math.round((p / 100) * 5)));
      setUploadPct(5);
      setProcessingStatus("Processing started...");

      let iv = null;
      const poll = async () => {
        try {
          const prog = await getMeetingProgress(meeting_id);
          setUploadPct(5 + Math.round((prog.progress ?? 0) * 0.95));
          setProcessingStatus(prog.message || prog.current_stage || "");
          if (prog.status === "completed") {
            clearInterval(iv);
            const data = await getMeeting(meeting_id);
            setSelectedMeeting(data);
            setMeetings((prev) => [data, ...prev.filter((m) => getMeetingId(m) !== meeting_id)]);
            setUploading(false);
            setTimeout(() => { setUploadPct(0); setProcessingStatus(""); }, 1200);
            loadAll(meeting_id);
          } else if (prog.status === "failed") {
            clearInterval(iv);
            setError(prog.message || "Processing failed");
            setUploading(false);
            setProcessingStatus("");
          }
        } catch (err) {
          clearInterval(iv);
          setError(err.response?.data?.detail || "Polling error");
          setUploading(false);
          setProcessingStatus("");
        }
      };
      iv = setInterval(poll, 1500);
      await poll();
    } catch (err) {
      setError(err.response?.data?.detail || "Upload failed");
      setUploading(false);
      setProcessingStatus("");
    }
  };

  const exportData = async (fmt) => {
    const id = getMeetingId(selectedMeeting);
    if (!id) return;
    try {
      const blob = await exportMeeting(id, fmt);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = `meeting-${id}.${fmt}`; a.click();
      URL.revokeObjectURL(url);
    } catch { setError("Export failed"); }
  };

  const runEvaluation = async () => {
    const id = getMeetingId(selectedMeeting);
    if (!id) return;
    setEvaluating(true);
    setEvalMsg("Running evaluation...");
    try {
      const res = await evaluateMeeting(id);
      setEvalMsg(`Evaluation completed: ${safeN(res.overall_quality_score, 1)}/100`);
      await loadAll(id);
    } catch (err) {
      setError(err.response?.data?.detail || "Evaluation failed");
    } finally {
      setEvaluating(false);
      setTimeout(() => setEvalMsg(""), 3000);
    }
  };

  const selectedId = getMeetingId(selectedMeeting);
  const meta = selectedMeeting?.metadata || {};
  const segments = selectedMeeting?.segments || [];
  const tasks = selectedMeeting?.tasks || [];
  const transcript = selectedMeeting?.meeting?.transcript || selectedMeeting?.transcript || "";
  const aliases = meta?.speaker_aliases || {};
  const latestEval = useMemo(
    () => evaluations.find((r) => r.meeting_id === selectedId) || null,
    [evaluations, selectedId]
  );

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ bgcolor: "background.default", minHeight: "100vh" }}>
        <Container maxWidth="xl" sx={{ py: 3 }}>
          {/* Header */}
          <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 3 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <Box sx={{ bgcolor: "primary.main", borderRadius: 2, p: 0.75, display: "flex" }}>
                <RecordVoiceOver sx={{ color: "#0d1117", fontSize: 22 }} />
              </Box>
              <Box>
                <Typography variant="h5" sx={{ fontWeight: 800, lineHeight: 1 }}>Meeting Secretary</Typography>
                <Typography variant="caption" color="text.secondary">AI transcription · task extraction · analytics</Typography>
              </Box>
            </Box>
            <Box sx={{ display: "flex", gap: 1 }}>
              <Button component="label" variant="contained" startIcon={<Upload />} disabled={uploading} size="small">
                Upload audio
                <input hidden type="file" accept="audio/*" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleUpload(f); e.target.value = ""; }} />
              </Button>
              <Button variant="outlined" startIcon={<Refresh />} size="small" onClick={() => loadAll(selectedId)} disabled={refreshing}>
                Refresh
              </Button>
            </Box>
          </Box>

          {uploading && (
            <Paper sx={{ p: 2, mb: 3 }}>
              <Typography variant="subtitle2" gutterBottom>{processingStatus || "Processing..."}</Typography>
              <LinearProgress variant="determinate" value={uploadPct} sx={{ mb: 0.5 }} />
              <Typography variant="caption" color="text.secondary">{uploadPct}%</Typography>
            </Paper>
          )}

          {evalMsg && <Alert severity="success" sx={{ mb: 3 }}>{evalMsg}</Alert>}

          {/* Top tabs */}
          <Box sx={{ borderBottom: 1, borderColor: "divider", mb: 3 }}>
            <Tabs value={topTab} onChange={(_, v) => setTopTab(v)}>
              <Tab label="Meetings" value="meetings" icon={<EventNote />} iconPosition="start" />
              <Tab label="Dashboard" value="dashboard" icon={<DashboardIcon />} iconPosition="start" />
            </Tabs>
          </Box>

          {/* Dashboard */}
          {topTab === "dashboard" && (
            <DashboardPage
              metrics={recentMetrics}
              evaluations={evaluations}
              meetings={meetings}
              stats={stats}
              onRefresh={() => loadAll(selectedId)}
              refreshing={refreshing}
            />
          )}

          {/* Meetings */}
          {topTab === "meetings" && (
            <Grid container spacing={3}>
              {/* Sidebar */}
              <Grid item xs={12} md={3}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="overline" sx={{ color: "text.secondary", fontSize: "0.65rem", letterSpacing: 1, fontWeight: 700 }}>
                    Meetings ({meetings.length})
                  </Typography>
                  <List dense sx={{ maxHeight: 680, overflow: "auto", mt: 1 }}>
                    {meetings.map((m) => {
                      const id = getMeetingId(m);
                      const status = m?.metadata?.status || "completed";
                      return (
                        <ListItem key={id} disablePadding sx={{ mb: 0.5 }}>
                          <ListItemButton selected={id === selectedId} onClick={() => { setSelectedMeeting(m); setMeetingTab("transcript"); }} sx={{ borderRadius: 1 }}>
                            <ListItemText
                              primary={
                                <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                                  <Typography variant="body2" sx={{ fontWeight: 700 }}>#{id}</Typography>
                                  <StatusDot status={status} />
                                </Box>
                              }
                              secondary={<Typography variant="caption" color="text.secondary">{formatDate(m?.meeting?.created_at || m?.metadata?.created_at)}</Typography>}
                            />
                          </ListItemButton>
                        </ListItem>
                      );
                    })}
                    {!meetings.length && <Typography variant="body2" color="text.secondary" sx={{ px: 1, py: 2, textAlign: "center" }}>No meetings yet</Typography>}
                  </List>
                </Paper>
              </Grid>

              {/* Main */}
              <Grid item xs={12} md={9}>
                {selectedMeeting ? (
                  <Paper sx={{ mb: 3 }}>
                    <Box sx={{ borderBottom: 1, borderColor: "divider", px: 2, py: 0.5, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <Tabs value={meetingTab} onChange={(_, v) => setMeetingTab(v)} variant="scrollable">
                        <Tab label="Transcript" value="transcript" icon={<Timeline />} iconPosition="start" />
                        <Tab label="Tasks" value="tasks" icon={<Assessment />} iconPosition="start" />
                        <Tab label="Metrics" value="metrics" icon={<BarChartIcon />} iconPosition="start" />
                      </Tabs>
                      <Box sx={{ display: "flex", gap: 0.5 }}>
                        {["json", "csv", "md", "txt"].map((fmt) => (
                          <MuiTooltip key={fmt} title={`Export ${fmt.toUpperCase()}`}>
                            <IconButton size="small" onClick={() => exportData(fmt)}>
                              <Download sx={{ fontSize: 16 }} />
                            </IconButton>
                          </MuiTooltip>
                        ))}
                      </Box>
                    </Box>
                    <Box sx={{ p: 3 }}>
                      {meetingTab === "transcript" && <TranscriptView segments={segments} transcript={transcript} metadata={meta} speakerAliases={aliases} />}
                      {meetingTab === "tasks" && <TasksView tasks={tasks} />}
                      {meetingTab === "metrics" && <MetricsView metadata={meta} latestEvaluation={latestEval} runEvaluation={runEvaluation} evaluating={evaluating} />}
                    </Box>
                  </Paper>
                ) : (
                  <Paper sx={{ p: 8, textAlign: "center" }}>
                    <RecordVoiceOver sx={{ fontSize: 56, color: "text.secondary", opacity: 0.3, mb: 2 }} />
                    <Typography variant="h6" color="text.secondary">Upload an audio file to get started</Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>Supports MP3, WAV, M4A, OGG and more</Typography>
                  </Paper>
                )}

                <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                  <Button variant="outlined" size="small" startIcon={<EventNote />} onClick={runEvaluation} disabled={!selectedMeeting || evaluating}>
                    {evaluating ? "Running..." : "Run evaluation"}
                  </Button>
                  <Button variant="outlined" size="small" startIcon={<Refresh />} onClick={() => loadAll(selectedId)} disabled={!selectedMeeting}>
                    Reload meeting
                  </Button>
                </Box>
              </Grid>
            </Grid>
          )}
        </Container>

        <Snackbar open={!!error} autoHideDuration={7000} onClose={() => setError(null)}>
          <Alert severity="error" onClose={() => setError(null)}>{error}</Alert>
        </Snackbar>
      </Box>
    </ThemeProvider>
  );
}
