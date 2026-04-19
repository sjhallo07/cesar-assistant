create table if not exists assistant_profiles (
    profile_id text primary key,
    display_name text,
    goal_summary text,
    master_prompt text,
    financial_objectives text,
    business_context text,
    constraints text,
    baseline_tone text,
    personality_summary text,
    personality_traits jsonb default '{}'::jsonb,
    rag_metadata jsonb default '{}'::jsonb,
    updated_at timestamptz default now()
);

create table if not exists assistant_conversations (
    id bigint generated always as identity primary key,
    profile_id text not null references assistant_profiles(profile_id) on delete cascade,
    chat_mode text not null,
    model_name text,
    user_message text not null,
    assistant_reply text not null,
    personality_summary text,
    personality_traits jsonb default '{}'::jsonb,
    data_rows_count integer default 0,
    created_at timestamptz default now()
);

create index if not exists idx_assistant_conversations_profile_id_created_at
    on assistant_conversations (profile_id, created_at desc);

alter table assistant_profiles enable row level security;
alter table assistant_conversations enable row level security;

drop policy if exists service_role_profiles_access on assistant_profiles;
create policy service_role_profiles_access
    on assistant_profiles
    for all
    using (true)
    with check (true);

drop policy if exists service_role_conversations_access on assistant_conversations;
create policy service_role_conversations_access
    on assistant_conversations
    for all
    using (true)
    with check (true);