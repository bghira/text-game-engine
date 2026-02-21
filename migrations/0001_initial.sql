BEGIN;

CREATE TABLE tge_actors (
  id TEXT PRIMARY KEY,
  display_name TEXT,
  kind TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL
);

CREATE TABLE tge_campaigns (
  id TEXT PRIMARY KEY,
  namespace TEXT NOT NULL DEFAULT 'default',
  name TEXT NOT NULL,
  name_normalized TEXT NOT NULL,
  created_by_actor_id TEXT REFERENCES tge_actors(id),
  summary TEXT NOT NULL DEFAULT '',
  state_json TEXT NOT NULL DEFAULT '{}',
  characters_json TEXT NOT NULL DEFAULT '{}',
  last_narration TEXT,
  memory_visible_max_turn_id BIGINT,
  row_version INTEGER NOT NULL DEFAULT 1,
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL,
  CONSTRAINT uq_tge_campaign_namespace_name_norm UNIQUE(namespace, name_normalized)
);

CREATE TABLE tge_sessions (
  id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL REFERENCES tge_campaigns(id),
  surface TEXT NOT NULL,
  surface_key TEXT NOT NULL UNIQUE,
  surface_guild_id TEXT,
  surface_channel_id TEXT,
  surface_thread_id TEXT,
  enabled BOOLEAN NOT NULL DEFAULT 1,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL
);

CREATE TABLE tge_actor_external_refs (
  id TEXT PRIMARY KEY,
  actor_id TEXT NOT NULL REFERENCES tge_actors(id),
  provider TEXT NOT NULL,
  external_id TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL,
  CONSTRAINT uq_tge_actor_external_ref_provider_external UNIQUE(provider, external_id),
  CONSTRAINT uq_tge_actor_external_ref_actor_provider_external UNIQUE(actor_id, provider, external_id)
);

CREATE TABLE tge_players (
  id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL REFERENCES tge_campaigns(id),
  actor_id TEXT NOT NULL REFERENCES tge_actors(id),
  level INTEGER NOT NULL DEFAULT 1,
  xp INTEGER NOT NULL DEFAULT 0,
  attributes_json TEXT NOT NULL DEFAULT '{}',
  state_json TEXT NOT NULL DEFAULT '{}',
  last_active_at TIMESTAMP,
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL,
  CONSTRAINT uq_tge_player_campaign_actor UNIQUE(campaign_id, actor_id)
);

CREATE TABLE tge_turns (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  campaign_id TEXT NOT NULL REFERENCES tge_campaigns(id),
  session_id TEXT REFERENCES tge_sessions(id),
  actor_id TEXT REFERENCES tge_actors(id),
  kind TEXT NOT NULL,
  content TEXT NOT NULL,
  meta_json TEXT NOT NULL DEFAULT '{}',
  external_message_id TEXT,
  external_user_message_id TEXT,
  created_at TIMESTAMP NOT NULL
);
CREATE INDEX ix_tge_turn_campaign_id_desc ON tge_turns(campaign_id, id DESC);
CREATE INDEX ix_tge_turn_campaign_external_msg ON tge_turns(campaign_id, external_message_id);

CREATE TABLE tge_snapshots (
  id TEXT PRIMARY KEY,
  turn_id INTEGER NOT NULL UNIQUE REFERENCES tge_turns(id),
  campaign_id TEXT NOT NULL REFERENCES tge_campaigns(id),
  campaign_state_json TEXT NOT NULL,
  campaign_characters_json TEXT NOT NULL,
  campaign_summary TEXT NOT NULL,
  campaign_last_narration TEXT,
  players_json TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL
);
CREATE INDEX ix_tge_snapshot_campaign_turn ON tge_snapshots(campaign_id, turn_id DESC);

CREATE TABLE tge_timers (
  id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL REFERENCES tge_campaigns(id),
  session_id TEXT REFERENCES tge_sessions(id),
  status TEXT NOT NULL,
  event_text TEXT NOT NULL,
  interruptible BOOLEAN NOT NULL DEFAULT 1,
  interrupt_action TEXT,
  due_at TIMESTAMP NOT NULL,
  fired_at TIMESTAMP,
  cancelled_at TIMESTAMP,
  external_message_id TEXT,
  external_channel_id TEXT,
  external_thread_id TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL,
  CONSTRAINT ck_tge_timers_timer_status_valid CHECK(status IN ('scheduled_unbound','scheduled_bound','cancelled','expired','consumed'))
);
CREATE INDEX ix_tge_timer_campaign_status_due ON tge_timers(campaign_id, status, due_at);
CREATE UNIQUE INDEX uq_tge_timer_one_active_per_campaign
ON tge_timers(campaign_id)
WHERE status IN ('scheduled_unbound','scheduled_bound');

CREATE TABLE tge_inflight_turns (
  id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL REFERENCES tge_campaigns(id),
  actor_id TEXT NOT NULL REFERENCES tge_actors(id),
  claim_token TEXT NOT NULL,
  claimed_at TIMESTAMP NOT NULL,
  heartbeat_at TIMESTAMP NOT NULL,
  expires_at TIMESTAMP NOT NULL,
  CONSTRAINT uq_tge_inflight_campaign_actor UNIQUE(campaign_id, actor_id)
);
CREATE INDEX ix_tge_inflight_expiry ON tge_inflight_turns(expires_at);

CREATE TABLE tge_media_refs (
  id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL REFERENCES tge_campaigns(id),
  player_id TEXT REFERENCES tge_players(id),
  ref_type TEXT NOT NULL,
  room_key TEXT,
  url TEXT NOT NULL,
  prompt TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL
);

CREATE TABLE tge_embeddings (
  turn_id INTEGER PRIMARY KEY REFERENCES tge_turns(id),
  campaign_id TEXT NOT NULL REFERENCES tge_campaigns(id),
  kind TEXT NOT NULL,
  content TEXT NOT NULL,
  embedding BLOB NOT NULL,
  created_at TIMESTAMP NOT NULL
);
CREATE INDEX ix_tge_embedding_campaign ON tge_embeddings(campaign_id);

CREATE TABLE tge_outbox_events (
  id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL REFERENCES tge_campaigns(id),
  session_id TEXT REFERENCES tge_sessions(id),
  session_scope TEXT NOT NULL DEFAULT '__none__',
  event_type TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0,
  next_attempt_at TIMESTAMP,
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL,
  CONSTRAINT uq_tge_outbox_campaign_session_event_key UNIQUE(campaign_id, session_scope, event_type, idempotency_key)
);
CREATE INDEX ix_tge_outbox_status_next_created ON tge_outbox_events(status, next_attempt_at, created_at);

COMMIT;
