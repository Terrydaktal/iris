use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Duration;

include!(concat!(env!("OUT_DIR"), "/iris_build_identity.rs"));

const DIAGNOSTIC_SCHEMA: u64 = 2;
const DEFAULT_TTL_SECONDS: u64 = 15 * 60;
const MAX_EVENT_COUNT: usize = 512;
const MAX_REQUEST_BYTES: u64 = 16 * 1024;
const MAX_RESPONSE_BYTES: usize = 1024 * 1024;

#[cfg(feature = "diagnostics")]
use std::collections::{HashMap, VecDeque};
#[cfg(feature = "diagnostics")]
use std::io::{Read, Write};
#[cfg(feature = "diagnostics")]
use std::os::unix::fs::PermissionsExt;
#[cfg(feature = "diagnostics")]
use std::os::unix::net::{UnixListener, UnixStream};
#[cfg(feature = "diagnostics")]
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
#[cfg(feature = "diagnostics")]
use std::sync::{Arc, Mutex, Weak};
#[cfg(feature = "diagnostics")]
use std::thread;
#[cfg(feature = "diagnostics")]
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone)]
pub(crate) struct DiagnosticState {
    #[cfg(feature = "diagnostics")]
    inner: Arc<DiagnosticInner>,
}

pub(crate) struct DiagnosticTaskGuard {
    #[cfg(feature = "diagnostics")]
    state: DiagnosticState,
    #[cfg(feature = "diagnostics")]
    task: String,
    #[cfg(feature = "diagnostics")]
    operation_id: u64,
    #[cfg(feature = "diagnostics")]
    completed: bool,
}

static GLOBAL_DIAGNOSTICS: OnceLock<DiagnosticState> = OnceLock::new();

#[cfg(feature = "diagnostics")]
struct DiagnosticInner {
    started_at: Instant,
    started_at_unix_ms: u64,
    enabled: AtomicBool,
    expires_at_ms: AtomicU64,
    shutdown: AtomicBool,
    heartbeat_ms: AtomicU64,
    frame_count: AtomicU64,
    background_active: AtomicU64,
    active_task_count: AtomicU64,
    active_task_high_water: AtomicU64,
    tasks_started: AtomicU64,
    tasks_completed: AtomicU64,
    tasks_failed: AtomicU64,
    thumbnail_active: AtomicU64,
    thumbnail_high_water: AtomicU64,
    background_high_water: AtomicU64,
    lock_wait_count: AtomicU64,
    lock_wait_total_us: AtomicU64,
    lock_wait_max_us: AtomicU64,
    lock_wait_last_us: AtomicU64,
    db_loaded: AtomicBool,
    grid_loading: AtomicBool,
    ui_state: AtomicU8,
    heartbeat_enabled: AtomicBool,
    events_enabled: AtomicBool,
    tasks_enabled: AtomicBool,
    next_operation_id: AtomicU64,
    next_event_sequence: AtomicU64,
    dropped_events: AtomicU64,
    task_registry_unavailable: AtomicBool,
    events: Mutex<VecDeque<DiagnosticEvent>>,
    active_tasks: Mutex<HashMap<u64, ActiveTask>>,
    last_failure: Mutex<Option<DiagnosticFailure>>,
}

#[cfg(feature = "diagnostics")]
#[derive(Clone)]
struct DiagnosticEvent {
    sequence: u64,
    monotonic_ms: u64,
    kind: String,
    operation_id: u64,
    state: String,
    code: String,
    reason: String,
}

#[cfg(feature = "diagnostics")]
struct ActiveTask {
    task: String,
    started_monotonic_ms: u64,
}

#[cfg(feature = "diagnostics")]
struct DiagnosticFailure {
    monotonic_ms: u64,
    task: String,
    operation_id: u64,
    code: String,
    reason: String,
}

impl DiagnosticState {
    pub(crate) fn new() -> Self {
        #[cfg(feature = "diagnostics")]
        {
            let now = Instant::now();
            let started_at_unix_ms = unix_time_ms();
            let enabled = env_flag("IRIS_DIAGNOSTICS");
            let heartbeat_enabled = env_flag_default("IRIS_DIAGNOSTICS_HEARTBEAT", true);
            let events_enabled = env_flag_default("IRIS_DIAGNOSTICS_EVENTS", true);
            let tasks_enabled = env_flag_default("IRIS_DIAGNOSTICS_TASKS", true);
            let ttl_seconds = std::env::var("IRIS_DIAGNOSTICS_TTL_SECS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(DEFAULT_TTL_SECONDS);
            let inner = Arc::new(DiagnosticInner {
                started_at: now,
                started_at_unix_ms,
                enabled: AtomicBool::new(enabled),
                expires_at_ms: AtomicU64::new(ttl_seconds.saturating_mul(1000)),
                shutdown: AtomicBool::new(false),
                heartbeat_ms: AtomicU64::new(0),
                frame_count: AtomicU64::new(0),
                background_active: AtomicU64::new(0),
                active_task_count: AtomicU64::new(0),
                active_task_high_water: AtomicU64::new(0),
                tasks_started: AtomicU64::new(0),
                tasks_completed: AtomicU64::new(0),
                tasks_failed: AtomicU64::new(0),
                thumbnail_active: AtomicU64::new(0),
                thumbnail_high_water: AtomicU64::new(0),
                background_high_water: AtomicU64::new(0),
                lock_wait_count: AtomicU64::new(0),
                lock_wait_total_us: AtomicU64::new(0),
                lock_wait_max_us: AtomicU64::new(0),
                lock_wait_last_us: AtomicU64::new(0),
                db_loaded: AtomicBool::new(false),
                grid_loading: AtomicBool::new(false),
                ui_state: AtomicU8::new(0),
                heartbeat_enabled: AtomicBool::new(heartbeat_enabled),
                events_enabled: AtomicBool::new(events_enabled),
                tasks_enabled: AtomicBool::new(tasks_enabled),
                next_operation_id: AtomicU64::new(1),
                next_event_sequence: AtomicU64::new(1),
                dropped_events: AtomicU64::new(0),
                task_registry_unavailable: AtomicBool::new(false),
                events: Mutex::new(VecDeque::with_capacity(MAX_EVENT_COUNT)),
                active_tasks: Mutex::new(HashMap::new()),
                last_failure: Mutex::new(None),
            });
            let state = Self {
                inner: inner.clone(),
            };
            // Keep only the tiny local control endpoint alive when collection is
            // disabled so a running observable build can be activated later.
            start_control_plane(&inner);
            state
        }

        #[cfg(not(feature = "diagnostics"))]
        {
            Self {}
        }
    }

    pub(crate) fn heartbeat(
        &self,
        ui_state: &str,
        background_active: usize,
        thumbnail_active: usize,
        db_loaded: bool,
        grid_loading: bool,
    ) {
        #[cfg(feature = "diagnostics")]
        {
            if !self.is_enabled() {
                return;
            }
            if !self.inner.heartbeat_enabled.load(Ordering::Relaxed) {
                return;
            }
            self.inner
                .heartbeat_ms
                .store(self.monotonic_ms(), Ordering::Relaxed);
            self.inner.frame_count.fetch_add(1, Ordering::Relaxed);
            self.inner
                .background_active
                .store(background_active as u64, Ordering::Relaxed);
            update_high_water(&self.inner.background_high_water, background_active as u64);
            self.inner
                .thumbnail_active
                .store(thumbnail_active as u64, Ordering::Relaxed);
            update_high_water(&self.inner.thumbnail_high_water, thumbnail_active as u64);
            self.inner.db_loaded.store(db_loaded, Ordering::Relaxed);
            self.inner
                .grid_loading
                .store(grid_loading, Ordering::Relaxed);
            self.inner
                .ui_state
                .store(ui_state_code(ui_state), Ordering::Relaxed);
        }

        #[cfg(not(feature = "diagnostics"))]
        {
            let _ = (
                ui_state,
                background_active,
                thumbnail_active,
                db_loaded,
                grid_loading,
            );
        }
    }

    pub(crate) fn next_operation_id(&self) -> u64 {
        #[cfg(feature = "diagnostics")]
        {
            return self.inner.next_operation_id.fetch_add(1, Ordering::Relaxed);
        }

        #[cfg(not(feature = "diagnostics"))]
        {
            0
        }
    }

    pub(crate) fn record_event(&self, kind: &str, operation_id: u64, state: &str, reason: &str) {
        self.record_event_with_code(kind, operation_id, state, "unspecified", reason);
    }

    pub(crate) fn record_event_with_code(
        &self,
        kind: &str,
        operation_id: u64,
        state: &str,
        code: &str,
        reason: &str,
    ) {
        #[cfg(feature = "diagnostics")]
        {
            if !self.is_enabled() || !self.inner.events_enabled.load(Ordering::Relaxed) {
                return;
            }
            let sequence = self
                .inner
                .next_event_sequence
                .fetch_add(1, Ordering::Relaxed);
            let event = DiagnosticEvent {
                sequence,
                monotonic_ms: self.monotonic_ms(),
                kind: bounded_string(kind),
                operation_id,
                state: bounded_string(state),
                code: bounded_string(code),
                reason: bounded_string(reason),
            };
            let Ok(mut events) = self.inner.events.try_lock() else {
                self.inner.dropped_events.fetch_add(1, Ordering::Relaxed);
                return;
            };
            if events.len() == MAX_EVENT_COUNT {
                events.pop_front();
                self.inner.dropped_events.fetch_add(1, Ordering::Relaxed);
            }
            events.push_back(event);
        }

        #[cfg(not(feature = "diagnostics"))]
        {
            let _ = (kind, operation_id, state, code, reason);
        }
    }

    pub(crate) fn task_started(&self, task: &str, operation_id: u64) {
        #[cfg(feature = "diagnostics")]
        {
            if self.is_enabled() {
                if self.inner.tasks_enabled.load(Ordering::Relaxed) {
                    self.inner.tasks_started.fetch_add(1, Ordering::Relaxed);
                    let registered = if let Ok(mut tasks) = self.inner.active_tasks.try_lock() {
                        tasks.insert(
                            operation_id,
                            ActiveTask {
                                task: bounded_string(task),
                                started_monotonic_ms: self.monotonic_ms(),
                            },
                        );
                        true
                    } else {
                        self.inner
                            .task_registry_unavailable
                            .store(true, Ordering::Release);
                        self.inner.dropped_events.fetch_add(1, Ordering::Relaxed);
                        false
                    };
                    if registered {
                        let active =
                            self.inner.active_task_count.fetch_add(1, Ordering::Relaxed) + 1;
                        update_high_water(&self.inner.active_task_high_water, active);
                    }
                }
                self.record_event_with_code(
                    "task_started",
                    operation_id,
                    task,
                    "task_started",
                    "created",
                );
            }
        }

        #[cfg(not(feature = "diagnostics"))]
        {
            let _ = (task, operation_id);
        }
    }

    pub(crate) fn task_completed(&self, task: &str, operation_id: u64) {
        #[cfg(feature = "diagnostics")]
        {
            let task_was_registered = match self.inner.active_tasks.try_lock() {
                Ok(mut tasks) => tasks.remove(&operation_id).is_some(),
                Err(_) => {
                    self.inner
                        .task_registry_unavailable
                        .store(true, Ordering::Release);
                    false
                }
            };
            if task_was_registered {
                self.inner.tasks_completed.fetch_add(1, Ordering::Relaxed);
                self.inner.active_task_count.fetch_sub(1, Ordering::Relaxed);
            }
            if self.is_enabled() {
                self.record_event_with_code(
                    "task_completed",
                    operation_id,
                    task,
                    "task_completed",
                    "completed",
                );
            }
        }

        #[cfg(not(feature = "diagnostics"))]
        {
            let _ = (task, operation_id);
        }
    }

    pub(crate) fn task_failed(&self, task: &str, operation_id: u64, reason: &str) {
        self.task_failed_with_code(task, operation_id, "worker_failure", reason);
    }

    pub(crate) fn task_failed_with_code(
        &self,
        task: &str,
        operation_id: u64,
        code: &str,
        reason: &str,
    ) {
        #[cfg(feature = "diagnostics")]
        {
            let task_was_registered = match self.inner.active_tasks.try_lock() {
                Ok(mut tasks) => tasks.remove(&operation_id).is_some(),
                Err(_) => {
                    self.inner
                        .task_registry_unavailable
                        .store(true, Ordering::Release);
                    false
                }
            };
            if task_was_registered {
                self.inner.tasks_failed.fetch_add(1, Ordering::Relaxed);
                self.inner.active_task_count.fetch_sub(1, Ordering::Relaxed);
            }
            if self.is_enabled() {
                self.record_event_with_code("task_failed", operation_id, task, code, reason);
                if let Ok(mut failure) = self.inner.last_failure.try_lock() {
                    *failure = Some(DiagnosticFailure {
                        monotonic_ms: self.monotonic_ms(),
                        task: bounded_string(task),
                        operation_id,
                        code: bounded_string(code),
                        reason: bounded_string(reason),
                    });
                }
                persist_last_failure(task, operation_id, code, reason);
            }
        }

        #[cfg(not(feature = "diagnostics"))]
        {
            let _ = (task, operation_id, code, reason);
        }
    }

    #[allow(dead_code)]
    pub(crate) fn record_lock_wait(&self, lock_name: &str, wait: Duration) {
        #[cfg(feature = "diagnostics")]
        {
            if !self.is_enabled() {
                return;
            }
            let wait_us = wait.as_micros().min(u64::MAX as u128) as u64;
            if wait_us == 0 {
                return;
            }
            self.inner.lock_wait_count.fetch_add(1, Ordering::Relaxed);
            self.inner
                .lock_wait_total_us
                .fetch_add(wait_us, Ordering::Relaxed);
            update_high_water(&self.inner.lock_wait_max_us, wait_us);
            self.inner
                .lock_wait_last_us
                .store(wait_us, Ordering::Relaxed);
            if wait_us >= 5_000 {
                self.record_event_with_code(
                    "lock_wait",
                    0,
                    lock_name,
                    "lock_contention",
                    &format!("wait_us={wait_us}"),
                );
            }
        }

        #[cfg(not(feature = "diagnostics"))]
        {
            let _ = (lock_name, wait);
        }
    }

    pub(crate) fn task_guard(&self, task: &str) -> DiagnosticTaskGuard {
        #[cfg(feature = "diagnostics")]
        {
            let operation_id = self.next_operation_id();
            self.task_started(task, operation_id);
            return DiagnosticTaskGuard {
                state: self.clone(),
                task: task.to_string(),
                operation_id,
                completed: false,
            };
        }

        #[cfg(not(feature = "diagnostics"))]
        {
            let _ = task;
            DiagnosticTaskGuard {}
        }
    }

    pub(crate) fn shutdown(&self) {
        #[cfg(feature = "diagnostics")]
        {
            self.inner.shutdown.store(true, Ordering::Release);
            self.inner.enabled.store(false, Ordering::Release);
            let _ = UnixStream::connect(diagnostic_socket_path());
        }
    }

    pub(crate) fn install_global(&self) {
        let _ = GLOBAL_DIAGNOSTICS.set(self.clone());
    }

    pub(crate) fn global() -> Option<Self> {
        GLOBAL_DIAGNOSTICS.get().cloned()
    }

    #[cfg(feature = "diagnostics")]
    fn is_enabled(&self) -> bool {
        let now = self.monotonic_ms();
        self.inner.enabled.load(Ordering::Acquire)
            && now < self.inner.expires_at_ms.load(Ordering::Acquire)
    }

    #[cfg(feature = "diagnostics")]
    fn monotonic_ms(&self) -> u64 {
        self.inner.started_at.elapsed().as_millis() as u64
    }
}

impl Default for DiagnosticState {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn build_info_value() -> Value {
    json!({
        "schema": DIAGNOSTIC_SCHEMA,
        "application": "iris",
        "package_version": env!("CARGO_PKG_VERSION"),
        "git_revision": GIT_REVISION,
        "git_dirty": GIT_DIRTY,
        "rustc": RUSTC_VERSION,
        "target": TARGET,
        "profile": PROFILE,
        "diagnostics_compiled": cfg!(feature = "diagnostics"),
    })
}

#[cfg(feature = "diagnostics")]
fn update_high_water(target: &AtomicU64, value: u64) {
    let mut current = target.load(Ordering::Relaxed);
    while value > current {
        match target.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

fn diagnostic_socket_path_for_pid(pid: u32) -> PathBuf {
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let user = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "default".to_string());
    let safe_user: String = user
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                character
            } else {
                '_'
            }
        })
        .collect();
    base.join(format!("iris-{safe_user}-{pid}.diagnostics.sock"))
}

pub(crate) fn diagnostic_socket_path() -> PathBuf {
    diagnostic_socket_path_for_pid(std::process::id())
}

pub(crate) fn handle_cli(args: &[String]) -> Option<Result<(), String>> {
    let command = args.get(1).map(String::as_str)?;
    match command {
        "--help" | "-h" => Some(Ok(println!(
            "Iris image and metadata viewer\n\nUsage: iris [OPTIONS] [PATH ...]\n\nOptions:\n  --same-window, -s       Open paths in the existing Iris window\n  --no-daemon             Keep the process attached to the terminal\n  --build-info            Print machine-readable build identity\n  --capabilities          Print available diagnostic capabilities\n  --diagnose COMMAND      Query IRIS_DIAGNOSTICS_PID\n  --diagnose-pid PID CMD  Query a specific Iris process\n  --version               Print the application version\n  --help                  Show this help"
        ))),
        "--version" => Some(Ok(println!(
            "iris {} ({})",
            env!("CARGO_PKG_VERSION"),
            GIT_REVISION
        ))),
        "--build-info" => Some(print_json(build_info_value())),
        "--capabilities" => Some(print_json(capabilities_value(false))),
        "--diagnose-pid" => {
            let pid = args
                .get(2)
                .ok_or_else(|| "--diagnose-pid requires a process ID".to_string())
                .and_then(|value| {
                    value
                        .parse::<u32>()
                        .map_err(|_| "--diagnose-pid requires a numeric process ID".to_string())
                });
            let request = args.get(3).map(String::as_str).unwrap_or("snapshot");
            let result = match pid {
                Ok(pid) => diagnostic_request(args, request, 4).map(|request| (request, Some(pid))),
                Err(error) => Err(error),
            };
            match result {
                Ok((request, pid)) => Some(request_socket(&request, pid)),
                Err(error) => Some(Err(error)),
            }
        }
        "--diagnose" => {
            let request = args.get(2).map(String::as_str).unwrap_or("snapshot");
            let pid = std::env::var("IRIS_DIAGNOSTICS_PID")
                .ok()
                .ok_or_else(|| {
                    "--diagnose requires --diagnose-pid PID or IRIS_DIAGNOSTICS_PID".to_string()
                })
                .and_then(|value| {
                    value
                        .parse::<u32>()
                        .map_err(|_| "IRIS_DIAGNOSTICS_PID must be numeric".to_string())
                });
            match (pid, diagnostic_request(args, request, 3)) {
                (Ok(pid), Ok(request)) => Some(request_socket(&request, Some(pid))),
                (Err(error), _) | (_, Err(error)) => Some(Err(error)),
            }
        }
        _ => None,
    }
}

fn diagnostic_request(
    args: &[String],
    request: &str,
    argument_offset: usize,
) -> Result<String, String> {
    let request = match request {
        "events" => {
            let limit = args
                .get(argument_offset)
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(100)
                .min(MAX_EVENT_COUNT);
            format!("events {limit}")
        }
        "activate" => {
            let ttl = args
                .get(argument_offset)
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(DEFAULT_TTL_SECONDS)
                .max(1)
                .min(24 * 60 * 60);
            format!("activate {ttl}")
        }
        "snapshot" | "capabilities" | "build-info" | "deactivate" => request.to_string(),
        "inject" => {
            let kind = args
                .get(argument_offset)
                .ok_or_else(|| "--diagnose inject requires a probe name".to_string())?;
            if kind != "task_failure" {
                return Err("only the task_failure diagnostic probe is supported".to_string());
            }
            let task = args
                .get(argument_offset + 1)
                .map(String::as_str)
                .unwrap_or("synthetic_probe");
            let code = args
                .get(argument_offset + 2)
                .map(String::as_str)
                .unwrap_or("synthetic_failure");
            let reason = args
                .get(argument_offset + 3)
                .map(String::as_str)
                .unwrap_or("diagnostic_fault_injection");
            format!("inject task_failure {task} {code} {reason}")
        }
        "set" => {
            let category = args
                .get(argument_offset)
                .ok_or_else(|| "--diagnose set requires a category".to_string());
            let value = args
                .get(argument_offset + 1)
                .ok_or_else(|| "--diagnose set requires on or off".to_string());
            match (category, value) {
                (Ok(category), Ok(value)) if matches!(value.as_str(), "on" | "off") => {
                    format!("set {category} {value}")
                }
                (Err(error), _) | (_, Err(error)) => return Err(error),
                _ => {
                    return Err("diagnostic category value must be on or off".to_string());
                }
            }
        }
        other => return Err(format!("unknown diagnostic command: {other}")),
    };
    Ok(request)
}

impl DiagnosticTaskGuard {
    #[cfg(feature = "diagnostics")]
    pub(crate) fn complete(mut self) {
        self.state.task_completed(&self.task, self.operation_id);
        self.completed = true;
    }

    #[cfg(not(feature = "diagnostics"))]
    pub(crate) fn complete(self) {}
}

impl Drop for DiagnosticTaskGuard {
    fn drop(&mut self) {
        #[cfg(feature = "diagnostics")]
        if !self.completed {
            self.state
                .task_failed(&self.task, self.operation_id, "task_dropped_or_panicked");
        }
    }
}

fn print_json(value: Value) -> Result<(), String> {
    serde_json::to_string_pretty(&value)
        .map(|output| println!("{output}"))
        .map_err(|error| format!("failed to serialize diagnostic response: {error}"))
}

fn capabilities_value(enabled: bool) -> Value {
    json!({
        "schema": DIAGNOSTIC_SCHEMA,
        "application": "iris",
        "diagnostics_compiled": cfg!(feature = "diagnostics"),
        "diagnostics_enabled": enabled,
        "commands": ["snapshot", "events [limit]", "activate [seconds]", "deactivate", "set CATEGORY on|off", "inject task_failure [task] [code] [reason]", "build-info", "capabilities"],
        "categories": ["heartbeat", "events", "tasks"],
        "socket": diagnostic_socket_path(),
        "limits": {
            "event_capacity": MAX_EVENT_COUNT,
            "max_request_bytes": MAX_REQUEST_BYTES,
            "max_response_bytes": MAX_RESPONSE_BYTES,
        },
    })
}

fn request_socket(request: &str, pid: Option<u32>) -> Result<(), String> {
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = (request, pid);
        return Err(
            "diagnostic control is not compiled into this release-minimal binary; rebuild with --features diagnostics"
                .to_string(),
        );
    }

    #[cfg(feature = "diagnostics")]
    {
        let path = pid
            .map(diagnostic_socket_path_for_pid)
            .unwrap_or_else(diagnostic_socket_path);
        let mut stream = UnixStream::connect(&path)
            .map_err(|error| format!("cannot connect to {}: {error}", path.display()))?;
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .map_err(|error| format!("cannot configure diagnostic socket: {error}"))?;
        stream
            .write_all(request.as_bytes())
            .map_err(|error| format!("cannot send diagnostic request: {error}"))?;
        stream
            .shutdown(std::net::Shutdown::Write)
            .map_err(|error| format!("cannot finish diagnostic request: {error}"))?;
        let mut response = Vec::new();
        stream
            .take(MAX_RESPONSE_BYTES as u64)
            .read_to_end(&mut response)
            .map_err(|error| format!("cannot read diagnostic response: {error}"))?;
        print_json(
            serde_json::from_slice(&response)
                .map_err(|error| format!("invalid diagnostic response: {error}"))?,
        )
    }
}

#[cfg(feature = "diagnostics")]
fn start_control_plane(inner: &Arc<DiagnosticInner>) {
    start_control_plane_at(inner, diagnostic_socket_path());
}

#[cfg(feature = "diagnostics")]
fn start_control_plane_at(inner: &Arc<DiagnosticInner>, path: PathBuf) {
    if path.exists() {
        if UnixStream::connect(&path).is_ok() {
            return;
        }
        let _ = std::fs::remove_file(&path);
    }
    let Ok(listener) = UnixListener::bind(&path) else {
        return;
    };
    if std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).is_err() {
        let _ = std::fs::remove_file(&path);
        return;
    }
    let weak = Arc::downgrade(inner);
    let _ = thread::Builder::new()
        .name("iris-diagnostics".to_string())
        .spawn(move || control_plane_loop(listener, weak, path));
}

#[cfg(feature = "diagnostics")]
fn control_plane_loop(listener: UnixListener, weak: Weak<DiagnosticInner>, path: PathBuf) {
    loop {
        let Some(inner) = weak.upgrade() else {
            break;
        };
        if inner.shutdown.load(Ordering::Acquire) {
            break;
        }
        match listener.accept() {
            Ok((mut stream, _)) => {
                if inner.shutdown.load(Ordering::Acquire) {
                    break;
                }
                let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
                let mut request = Vec::new();
                let _ = std::io::Read::by_ref(&mut stream)
                    .take(MAX_REQUEST_BYTES + 1)
                    .read_to_end(&mut request);
                let response = if request.len() <= MAX_REQUEST_BYTES as usize {
                    String::from_utf8(request)
                        .map(|request| handle_request(&inner, request.trim()))
                        .unwrap_or_else(|_| json_error("request is not valid UTF-8"))
                } else {
                    json_error("request exceeds the diagnostic size limit")
                };
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.shutdown(std::net::Shutdown::Both);
            }
            Err(_) if inner.shutdown.load(Ordering::Acquire) => break,
            Err(_) => thread::sleep(Duration::from_millis(100)),
        }
    }
    let _ = std::fs::remove_file(path);
}

#[cfg(feature = "diagnostics")]
fn handle_request(inner: &Arc<DiagnosticInner>, request: &str) -> String {
    let mut parts = request.split_whitespace();
    let command = parts.next().unwrap_or("snapshot");
    let value = match command {
        "snapshot" => snapshot_value(inner),
        "events" => {
            let limit = parts
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(100)
                .min(MAX_EVENT_COUNT);
            events_value(inner, limit)
        }
        "capabilities" => capabilities_value(is_enabled(inner)),
        "build-info" => build_info_value(),
        "activate" => {
            let ttl = parts
                .next()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(DEFAULT_TTL_SECONDS)
                .max(1)
                .min(24 * 60 * 60);
            inner.enabled.store(true, Ordering::Release);
            inner.expires_at_ms.store(
                monotonic_ms(inner).saturating_add(ttl.saturating_mul(1000)),
                Ordering::Release,
            );
            record_configuration_event(inner, "diagnostics", "activate", &format!("ttl={ttl}s"));
            snapshot_value(inner)
        }
        "deactivate" => {
            record_configuration_event(inner, "diagnostics", "deactivate", "disabled");
            inner.enabled.store(false, Ordering::Release);
            capabilities_value(false)
        }
        "set" => {
            let category = parts.next().unwrap_or_default();
            let enabled = matches!(parts.next(), Some("on"));
            if !set_category(inner, category, enabled) {
                return json_error("unknown diagnostic category");
            }
            record_configuration_event(inner, category, "set", if enabled { "on" } else { "off" });
            capabilities_value(is_enabled(inner))
        }
        "inject" => match parts.next() {
            Some("task_failure") => {
                let operation_id = inner.next_operation_id.fetch_add(1, Ordering::Relaxed);
                let task = parts.next().unwrap_or("synthetic_probe");
                let code = parts.next().unwrap_or("synthetic_failure");
                let reason = parts.next().unwrap_or("diagnostic_fault_injection");
                let event = DiagnosticEvent {
                    sequence: inner.next_event_sequence.fetch_add(1, Ordering::Relaxed),
                    monotonic_ms: monotonic_ms(inner),
                    kind: "injected_failure".to_string(),
                    operation_id,
                    state: bounded_string(task),
                    code: bounded_string(code),
                    reason: bounded_string(reason),
                };
                if let Ok(mut events) = inner.events.try_lock() {
                    if events.len() == MAX_EVENT_COUNT {
                        events.pop_front();
                        inner.dropped_events.fetch_add(1, Ordering::Relaxed);
                    }
                    events.push_back(event);
                } else {
                    inner.dropped_events.fetch_add(1, Ordering::Relaxed);
                }
                json!({
                    "schema": DIAGNOSTIC_SCHEMA,
                    "injected": true,
                    "operation_id": operation_id,
                })
            }
            _ => json!({
                "schema": DIAGNOSTIC_SCHEMA,
                "error": "inject requires task_failure [task] [code] [reason]",
            }),
        },
        _ => json!({
            "schema": DIAGNOSTIC_SCHEMA,
            "error": "unknown diagnostic command",
        }),
    };
    serde_json::to_string(&value).unwrap_or_else(|_| json_error("failed to serialize response"))
}

#[cfg(feature = "diagnostics")]
fn snapshot_value(inner: &Arc<DiagnosticInner>) -> Value {
    let now = monotonic_ms(inner);
    let (active_task_details, task_registry_unavailable) = active_task_values(inner);
    let unavailable_subsystems =
        if task_registry_unavailable || inner.task_registry_unavailable.load(Ordering::Acquire) {
            vec!["task_registry"]
        } else {
            Vec::new()
        };
    let last_failure = inner.last_failure.try_lock().ok().and_then(|failure| {
        failure.as_ref().map(|failure| {
            json!({
                "monotonic_ms": failure.monotonic_ms,
                "task": failure.task,
                "operation_id": failure.operation_id,
                "code": failure.code,
                "reason": failure.reason,
            })
        })
    });
    json!({
        "schema": DIAGNOSTIC_SCHEMA,
        "consistency": "best-effort",
        "captured_at_unix_ms": unix_time_ms(),
        "capture_monotonic_ms": now,
        "build": build_info_value(),
        "process": {
            "pid": std::process::id(),
            "start_unix_ms": inner.started_at_unix_ms,
            "resource_evidence": process_resource_evidence(),
        },
        "diagnostics": {
            "enabled": is_enabled(inner),
            "expires_in_ms": inner.expires_at_ms.load(Ordering::Acquire).saturating_sub(now),
            "categories": {
                "heartbeat": inner.heartbeat_enabled.load(Ordering::Relaxed),
                "events": inner.events_enabled.load(Ordering::Relaxed),
                "tasks": inner.tasks_enabled.load(Ordering::Relaxed),
            },
            "heartbeat_age_ms": now.saturating_sub(inner.heartbeat_ms.load(Ordering::Acquire)),
            "frame_count": inner.frame_count.load(Ordering::Relaxed),
            "ui_state": ui_state_name(inner.ui_state.load(Ordering::Relaxed)),
            "background_active": inner.background_active.load(Ordering::Relaxed),
            "active_tasks": inner.active_task_count.load(Ordering::Relaxed),
            "active_task_high_water": inner.active_task_high_water.load(Ordering::Relaxed),
            "active_task_details": active_task_details,
            "tasks_started": inner.tasks_started.load(Ordering::Relaxed),
            "tasks_completed": inner.tasks_completed.load(Ordering::Relaxed),
            "tasks_failed": inner.tasks_failed.load(Ordering::Relaxed),
            "thumbnail_active": inner.thumbnail_active.load(Ordering::Relaxed),
            "resource_high_water": {
                "background_tasks": inner.background_high_water.load(Ordering::Relaxed),
                "thumbnail_tasks": inner.thumbnail_high_water.load(Ordering::Relaxed),
            },
            "lock_waits": {
                "count": inner.lock_wait_count.load(Ordering::Relaxed),
                "total_us": inner.lock_wait_total_us.load(Ordering::Relaxed),
                "max_us": inner.lock_wait_max_us.load(Ordering::Relaxed),
                "last_us": inner.lock_wait_last_us.load(Ordering::Relaxed),
                "sample_threshold_us": 5000,
            },
            "db_loaded": inner.db_loaded.load(Ordering::Relaxed),
            "grid_loading": inner.grid_loading.load(Ordering::Relaxed),
            "last_failure": last_failure,
        },
        "unavailable_subsystems": unavailable_subsystems,
    })
}

#[cfg(feature = "diagnostics")]
fn events_value(inner: &Arc<DiagnosticInner>, limit: usize) -> Value {
    let Ok(events) = inner.events.try_lock() else {
        return json!({
            "schema": DIAGNOSTIC_SCHEMA,
            "consistency": "best-effort",
            "unavailable_subsystems": ["event_ring"],
            "dropped_events": inner.dropped_events.load(Ordering::Relaxed),
        });
    };
    let start = events.len().saturating_sub(limit);
    let values = events
        .iter()
        .skip(start)
        .map(|event| {
            json!({
                "sequence": event.sequence,
                "monotonic_ms": event.monotonic_ms,
                "kind": event.kind,
                "operation_id": event.operation_id,
                "state": event.state,
                "code": event.code,
                "reason": event.reason,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "schema": DIAGNOSTIC_SCHEMA,
        "consistency": "best-effort",
        "events": values,
        "capacity": MAX_EVENT_COUNT,
        "dropped_or_overwritten": inner.dropped_events.load(Ordering::Relaxed),
        "oldest_sequence": events.front().map(|event| event.sequence),
        "newest_sequence": events.back().map(|event| event.sequence),
    })
}

#[cfg(feature = "diagnostics")]
fn is_enabled(inner: &DiagnosticInner) -> bool {
    let now = monotonic_ms(inner);
    inner.enabled.load(Ordering::Acquire) && now < inner.expires_at_ms.load(Ordering::Acquire)
}

#[cfg(feature = "diagnostics")]
fn monotonic_ms(inner: &DiagnosticInner) -> u64 {
    inner.started_at.elapsed().as_millis() as u64
}

#[cfg(feature = "diagnostics")]
fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("on")
    )
}

#[cfg(feature = "diagnostics")]
fn env_flag_default(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(default)
}

#[cfg(feature = "diagnostics")]
fn set_category(inner: &DiagnosticInner, category: &str, enabled: bool) -> bool {
    match category {
        "heartbeat" => inner.heartbeat_enabled.store(enabled, Ordering::Release),
        "events" => inner.events_enabled.store(enabled, Ordering::Release),
        "tasks" => inner.tasks_enabled.store(enabled, Ordering::Release),
        _ => return false,
    }
    true
}

#[cfg(feature = "diagnostics")]
fn record_configuration_event(inner: &DiagnosticInner, category: &str, action: &str, value: &str) {
    let sequence = inner.next_event_sequence.fetch_add(1, Ordering::Relaxed);
    let event = DiagnosticEvent {
        sequence,
        monotonic_ms: monotonic_ms(inner),
        kind: "configuration".to_string(),
        operation_id: 0,
        state: bounded_string(category),
        code: bounded_string(action),
        reason: bounded_string(value),
    };
    let Ok(mut events) = inner.events.try_lock() else {
        inner.dropped_events.fetch_add(1, Ordering::Relaxed);
        return;
    };
    if events.len() == MAX_EVENT_COUNT {
        events.pop_front();
        inner.dropped_events.fetch_add(1, Ordering::Relaxed);
    }
    events.push_back(event);
}

#[cfg(feature = "diagnostics")]
fn process_resource_evidence() -> Value {
    let mut values = serde_json::Map::new();
    for (key, path) in [
        ("oom_score", "/proc/self/oom_score"),
        ("oom_score_adj", "/proc/self/oom_score_adj"),
        ("vm_rss_kb", "/proc/self/status"),
    ] {
        if let Some(contents) = read_bounded_file(path, 4096) {
            if key == "vm_rss_kb" {
                let value = contents
                    .lines()
                    .find(|line| line.starts_with("VmRSS:"))
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                values.insert(key.to_string(), Value::String(value));
            } else {
                values.insert(key.to_string(), Value::String(contents.trim().to_string()));
            }
        }
    }
    if let Some((cgroup_path, memory_events)) = cgroup_memory_events() {
        values.insert("cgroup_path".to_string(), Value::String(cgroup_path));
        values.insert(
            "cgroup_memory_events".to_string(),
            Value::String(memory_events),
        );
    }
    Value::Object(values)
}

#[cfg(feature = "diagnostics")]
fn read_bounded_file(path: &str, max_bytes: u64) -> Option<String> {
    let file = std::fs::File::open(path).ok()?;
    let mut contents = String::new();
    file.take(max_bytes)
        .read_to_string(&mut contents)
        .ok()
        .map(|_| contents)
}

#[cfg(feature = "diagnostics")]
fn cgroup_memory_events() -> Option<(String, String)> {
    let cgroups = read_bounded_file("/proc/self/cgroup", 16 * 1024)?;
    for line in cgroups.lines() {
        let mut fields = line.splitn(3, ':');
        let _hierarchy = fields.next()?;
        let controllers = fields.next()?;
        let relative = fields.next()?.trim();
        if !controllers.is_empty()
            && !controllers
                .split(',')
                .any(|controller| controller == "memory")
        {
            continue;
        }
        if relative.split('/').any(|component| component == "..") {
            continue;
        }
        let relative = relative.trim_start_matches('/');
        let cgroup_root = PathBuf::from("/sys/fs/cgroup");
        let events_path = cgroup_root.join(relative).join("memory.events");
        if let Some(events) = read_bounded_file(events_path.to_string_lossy().as_ref(), 4096) {
            let display_path = if relative.is_empty() {
                "/".to_string()
            } else {
                format!("/{relative}")
            };
            return Some((display_path, events.trim().to_string()));
        }
    }
    None
}

#[cfg(feature = "diagnostics")]
fn persist_last_failure(task: &str, operation_id: u64, code: &str, reason: &str) {
    let Ok(path) = std::env::var("IRIS_DIAGNOSTICS_FAILURE_FILE") else {
        return;
    };
    if path.trim().is_empty() {
        return;
    }
    let value = json!({
        "schema": DIAGNOSTIC_SCHEMA,
        "captured_at_unix_ms": unix_time_ms(),
        "process": { "pid": std::process::id() },
        "build": build_info_value(),
        "task": bounded_string(task),
        "operation_id": operation_id,
        "code": bounded_string(code),
        "reason": bounded_string(reason),
    });
    let Ok(serialized) = serde_json::to_vec_pretty(&value) else {
        return;
    };
    let target = PathBuf::from(path);
    let temporary = target.with_extension(format!("{}.tmp", std::process::id()));
    if let Some(parent) = target.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::fs::write(&temporary, serialized).is_ok() {
        let _ = std::fs::rename(temporary, target);
    }
}

#[cfg(feature = "diagnostics")]
fn active_task_values(inner: &DiagnosticInner) -> (Vec<Value>, bool) {
    let Ok(tasks) = inner.active_tasks.try_lock() else {
        return (Vec::new(), true);
    };
    let now = monotonic_ms(inner);
    let values = tasks
        .iter()
        .map(|(operation_id, task)| {
            json!({
                "operation_id": operation_id,
                "task": task.task,
                "age_ms": now.saturating_sub(task.started_monotonic_ms),
            })
        })
        .collect();
    (values, false)
}

#[cfg(feature = "diagnostics")]
fn bounded_string(value: &str) -> String {
    value.chars().take(96).collect()
}

#[cfg(feature = "diagnostics")]
fn json_error(message: &str) -> String {
    serde_json::to_string(&json!({
        "schema": DIAGNOSTIC_SCHEMA,
        "error": message,
    }))
    .unwrap_or_else(|_| "{\"error\":\"diagnostic failure\"}".to_string())
}

#[cfg(feature = "diagnostics")]
fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

#[cfg(feature = "diagnostics")]
fn ui_state_code(value: &str) -> u8 {
    match value {
        "home" => 1,
        "grid" => 2,
        "viewer" => 3,
        "comparison" => 4,
        _ => 0,
    }
}

#[cfg(feature = "diagnostics")]
fn ui_state_name(value: u8) -> &'static str {
    match value {
        1 => "home",
        2 => "grid",
        3 => "viewer",
        4 => "comparison",
        _ => "unknown",
    }
}

#[cfg(all(test, feature = "diagnostics"))]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_ring_is_bounded_and_reports_overwrites() {
        let state = DiagnosticState::new();
        state.inner.enabled.store(true, Ordering::Release);
        state.inner.expires_at_ms.store(u64::MAX, Ordering::Release);

        for sequence in 0..(MAX_EVENT_COUNT + 3) {
            state.record_event("test", sequence as u64, "running", "bounded");
        }

        let events = events_value(&state.inner, MAX_EVENT_COUNT + 10);
        assert_eq!(events["capacity"], MAX_EVENT_COUNT);
        assert_eq!(
            events["events"].as_array().map(Vec::len),
            Some(MAX_EVENT_COUNT)
        );
        assert_eq!(events["dropped_or_overwritten"], 3);
    }

    #[test]
    fn snapshot_reports_active_task_identity_without_media_paths() {
        let state = DiagnosticState::new();
        state.inner.enabled.store(true, Ordering::Release);
        state.inner.expires_at_ms.store(u64::MAX, Ordering::Release);
        let operation_id = state.next_operation_id();
        state.task_started("metadata_worker", operation_id);

        let snapshot = snapshot_value(&state.inner);
        assert_eq!(snapshot["diagnostics"]["active_tasks"], 1);
        assert_eq!(
            snapshot["diagnostics"]["active_task_details"][0]["task"],
            "metadata_worker"
        );
        assert!(
            snapshot["diagnostics"]
                .to_string()
                .find("/media/")
                .is_none()
        );
    }

    #[test]
    fn diagnostic_socket_is_scoped_to_the_process() {
        let path = diagnostic_socket_path();
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or_default();
        assert!(name.ends_with(&format!("-{}.diagnostics.sock", std::process::id())));
    }

    #[test]
    fn task_guard_records_completion_without_leaking_active_tasks() {
        let state = DiagnosticState::new();
        state.inner.enabled.store(true, Ordering::Release);
        state.inner.expires_at_ms.store(u64::MAX, Ordering::Release);
        state.task_guard("test_worker").complete();

        let snapshot = snapshot_value(&state.inner);
        assert_eq!(snapshot["diagnostics"]["active_tasks"], 0);
        assert_eq!(snapshot["diagnostics"]["tasks_completed"], 1);
    }

    #[test]
    fn lock_waits_and_failure_codes_are_retained() {
        let state = DiagnosticState::new();
        state.inner.enabled.store(true, Ordering::Release);
        state.inner.expires_at_ms.store(u64::MAX, Ordering::Release);
        state.record_lock_wait("test_lock", Duration::from_millis(6));
        let operation_id = state.next_operation_id();
        state.task_started("test_worker", operation_id);
        state.task_failed_with_code(
            "test_worker",
            operation_id,
            "test_failure",
            "synthetic reason",
        );

        let snapshot = snapshot_value(&state.inner);
        assert_eq!(snapshot["diagnostics"]["lock_waits"]["count"], 1);
        assert_eq!(
            snapshot["diagnostics"]["last_failure"]["code"],
            "test_failure"
        );
        let events = events_value(&state.inner, MAX_EVENT_COUNT);
        assert!(events["events"].as_array().is_some_and(|events| {
            events
                .iter()
                .any(|event| event["kind"] == "lock_wait" && event["code"] == "lock_contention")
        }));
    }

    #[test]
    fn control_plane_can_activate_runtime_disabled_state() {
        let state = DiagnosticState::new();
        state.inner.enabled.store(false, Ordering::Release);
        let path = std::env::temp_dir().join(format!(
            "iris-diagnostics-test-{}-{}.sock",
            std::process::id(),
            state.next_operation_id()
        ));
        let _ = std::fs::remove_file(&path);
        start_control_plane_at(&state.inner, path.clone());

        let mut response = String::new();
        for _ in 0..50 {
            if let Ok(mut stream) = UnixStream::connect(&path) {
                stream.write_all(b"activate 30").unwrap();
                stream.shutdown(std::net::Shutdown::Write).unwrap();
                stream.read_to_string(&mut response).unwrap();
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        assert!(response.contains("\"enabled\":true"));
        assert!(state.is_enabled());
        state.shutdown();
        let _ = UnixStream::connect(&path);
        let _ = std::fs::remove_file(path);
    }
}
