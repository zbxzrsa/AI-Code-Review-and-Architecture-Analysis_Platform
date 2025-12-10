proposal

## Chapter 1 Conclusion: High-Level System Architecture

Alt-text: Overview of platform components, data flow, boundaries, and interactions.

This diagram references major subsystems (Frontend, Gateway, AI services, Data stores, Monitoring) and their data flows and boundaries.

```plantuml
@startuml
skinparam backgroundColor #FFFFFF
skinparam defaultFontName Calibri
skinparam ArrowColor #334155
skinparam NoteBackgroundColor #F8FAFC
skinparam NoteBorderColor #CBD5E1
skinparam componentBorderColor #4F46E5
skinparam componentBackgroundColor #EEF2FF
skinparam rectangleBorderColor #0EA5E9
skinparam rectangleBackgroundColor #E0F2FE

rectangle "User Boundary" as UB {
  actor User as user
}

rectangle "Edge/Gateway Boundary" as GB {
  component "API Gateway" as api
  component "Ingress (TLS)" as ingress
}

rectangle "Application Boundary" as AB {
  component "Frontend (React/SPA)" as fe
  component "Backend API (FastAPI)" as be
  component "Code Review AI (CRAI)" as crai
  component "Version Control AI (VCAI)" as vcai
}

rectangle "Data Boundary" as DB {
  database "PostgreSQL" as pg
  collections "Redis" as redis
  queue "Kafka" as kafka
  storage "Object Storage (S3)" as s3
}

rectangle "Observability Boundary" as OB {
  component "Prometheus" as prom
  component "Grafana" as graf
  component "OTel Collector" as otel
}

user --> ingress : HTTPS
ingress --> api : mTLS
api --> fe : WebSocket/HTTP
fe --> be : JSON/REST
be --> crai : gRPC
be --> vcai : gRPC
crai --> pg : SQL
crai --> redis : Cache
crai --> s3 : Artifacts
vcai --> pg : SQL
vcai --> kafka : Events
be --> kafka : Events
otel ..> be : OTLP
otel ..> crai : OTLP
prom ..> be : /metrics
graf ..> prom : queries

legend left
  Components: Frontend, Backend, AI services, Data stores, Observability
  Flows: HTTPS/mTLS, REST, gRPC, SQL, OTLP, Events
  Boundaries: User, Edge/Gateway, Application, Data, Observability
endlegend
@enduml
```

## Chapter 3 Development Tools Section: Technical Stack Architecture

Alt-text: Stack layers with front-end integration, AI components, data layer, and protocols.

Referenced stack: UI, API, AI services, Data stores, and Observability tools.

```plantuml
@startuml
skinparam backgroundColor #FFFFFF
skinparam defaultFontName Calibri
skinparam ArrowColor #334155
skinparam componentBorderColor #4F46E5
skinparam componentBackgroundColor #EEF2FF

package "UI Layer" {
  component "React + TypeScript" as react
  component "State (Redux/Query)" as state
}

package "API Layer" {
  component "FastAPI" as fastapi
  component "Auth (JWT/OAuth2)" as auth
}

package "AI Services" {
  component "CRAI Service" as crai
  component "VCAI Service" as vcai
}

package "Data Layer" {
  database "PostgreSQL" as pg
  collections "Redis" as redis
  queue "Kafka" as kafka
  storage "S3" as s3
}

package "Observability" {
  component "OpenTelemetry" as otel
  component "Prometheus" as prom
}

react --> fastapi : REST/JSON
state --> fastapi : WebSocket
fastapi --> auth : OAuth2/JWT
fastapi --> crai : gRPC
fastapi --> vcai : gRPC
crai --> pg : SQL
crai --> redis : Cache
vcai --> kafka : Producer
vcai --> s3 : Artifacts
fastapi --> prom : /metrics
otel ..> fastapi : OTLP
otel ..> crai : OTLP

legend left
  Protocols: REST/JSON, WebSocket, gRPC, SQL, OTLP
  Layers: UI, API, AI Services, Data, Observability
endlegend
@enduml
```

## Chapter 6.4: C4 Level 1 System Context

Alt-text: System context showing primary actors, core system, and external dependencies.

System context references: User, Code Review Platform, External SCM/OAuth providers.

```plantuml
@startuml
skinparam backgroundColor #FFFFFF
skinparam defaultFontName Calibri
skinparam ArrowColor #334155
skinparam rectangleBorderColor #0EA5E9
skinparam rectangleBackgroundColor #E0F2FE

actor "Developer" as dev
actor "Admin" as admin

rectangle "Code Review Platform" as platform {
  component "Frontend" as c4_fe
  component "Backend API" as c4_api
  component "AI Services" as c4_ai
}

rectangle "External Systems" as ext {
  component "Git Provider (GitHub/GitLab)" as scm
  component "OAuth2 IdP" as idp
}

dev --> c4_fe : uses
admin --> c4_fe : configures
c4_fe --> c4_api : interacts
c4_api --> c4_ai : requests analysis
c4_api --> scm : fetches repos/PRs
c4_api --> idp : authenticates (OAuth2)

note right of platform
  Context: Code Review AI evaluates code and reports
  Interactions: UI->API->AI Services; API->External Providers
end note

legend left
  C4 Level 1: System Context
  Persons: Developer, Admin
  Systems: Platform, External SCM, OAuth IdP
  Relationships: use, configure, authenticate, analyze
endlegend
@enduml
```

## Chapter 6.5: Agile Development Flowchart

Alt-text: Flowchart of iterative sprints, QA gates, and deployment stages.

This flowchart references sprint planning, development, QA, and deployment.

```plantuml
@startuml
skinparam backgroundColor #FFFFFF
skinparam defaultFontName Calibri
skinparam ArrowColor #334155
skinparam activityBorderColor #4F46E5
skinparam activityBackgroundColor #EEF2FF

start
:Backlog Grooming;
:Sprint Planning;
repeat
:Development & Code Review;
if (Feature Complete?) then (yes)
  :Unit/Integration Tests;
  if (QA Pass?) then (yes)
    :Staging Deploy;
    :UAT;
    if (UAT Approved?) then (yes)
      :Production Deploy;
      :Monitor & Collect Metrics;
    else (no)
      :Fix Findings;
    endif
  else (no)
    :Bug Fixes & Re-test;
  endif
else (no)
  :Continue Implementation;
endif
repeat while (Sprint Timebox)
:Sprint Review & Retrospective;
stop

legend left
  Iterations: repeat while sprint timebox
  QA Checkpoints: Unit/Integration, QA, UAT
  Stages: Staging, Production, Monitoring
endlegend
@enduml
```

## Chapter 6.6: Functional Interface Schematics

Alt-text: UI layout, navigation, interactions, and state transitions.

Schematics reference: Dashboard, Project Detail, Analysis Report, Settings.

```plantuml
@startuml
skinparam backgroundColor #FFFFFF
skinparam defaultFontName Calibri
skinparam ArrowColor #334155

salt
{
  "Dashboard"
  [Projects] [Analyses] [Settings]
}

salt
{
  "Project Detail"
  [Repo Info] [Open PRs] [Run Analysis]
}

salt
{
  "Analysis Report"
  [Issues List] [Summary] [Export]
}

salt
{
  "Settings"
  [Providers] [API Keys] [Roles]
}

"Dashboard" --> "Project Detail" : Navigate to project
"Project Detail" --> "Analysis Report" : View results
"Settings" --> "Providers" : Configure integrations

legend left
  UI Components: Dashboard, Project, Report, Settings
  Navigation: Dashboard -> Project -> Report
  Interactions: Run analysis, export, configure
endlegend
@enduml
```

## Chapter 6.7: User Journey Map

Alt-text: Journey from entry to completion with decision and pain points.

Referenced journey: Entry -> Explore -> Analyze -> Review -> Act.

```plantuml
@startmindmap
skinparam backgroundColor #FFFFFF
skinparam defaultFontName Calibri

* User Journey
** Entry
*** Landing Page
*** Sign In
** Explore
*** Browse Projects
*** Connect Repository
** Analyze
*** Start Analysis
*** View Progress
** Review
*** Inspect Findings
*** Prioritize Issues
** Act
*** Apply Fixes
*** Verify & Deploy

legend left
  Decision Points: Connect repo, approve fixes
  Completion Criteria: Findings resolved, deployment successful
  Pain Points: Setup friction, false positives, long runtimes
endlegend
@endmindmap
```

