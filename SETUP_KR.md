# Claude Code Research Lab 템플릿 적용 방법

## 1) 복사
이 템플릿의 파일/디렉토리를 **프로젝트 repo 루트**로 복사하세요:
- `CLAUDE.md`
- `.claude/` (settings, rules, agents, skills, hooks)
- `notes/`, `experiments/`, `reports/`, `scripts/`

## 2) .gitignore 업데이트
`gitignore.additions` 내용을 기존 `.gitignore`에 append 권장:
- `.claude/worktrees/` 를 ignore 하지 않으면 worktree 전체가 untracked로 잡힐 수 있습니다.

## 3) 로컬 설정
`.claude/settings.local.example.json` → `.claude/settings.local.json` 로 복사한 뒤 필요에 맞게 수정하세요.
(이 파일은 gitignore 대상)

### 비용/토큰 절감 관련 기본값
- `MAX_THINKING_TOKENS`: 기본 8000 (가벼운 작업에서 비용 절감). 플랜/설계 작업에서 품질이 필요하면 더 올리세요.
- `ENABLE_TOOL_SEARCH`: `auto:5` (MCP tool 정의가 컨텍스트를 많이 먹을 때 on-demand 로딩을 더 빨리 켬)

### verbose output 슬리밍 우회
- 큰 로그를 그대로 보고 싶을 때: `CC_FULL_OUTPUT=1` 환경변수를 세션에 설정

## 4) 병렬(worktree) 운영
Claude Code는 다음으로 **격리된 git worktree**에서 실행할 수 있습니다:
- `claude -w za`
- `claude -w zb`
- `claude -w report-v1`

Worktree 경로:
- `<repo>/.claude/worktrees/<name>`

수동 관리가 필요하면:
- `scripts/cc-wt new za`
- `cd "$(scripts/cc-wt path za)"`

## 5) 권장 루프
- 복잡한 작업: **Plan mode → PlanReviewer 리뷰 → 구현/실험 → /verify → /retro**
- 자주 반복되는 작업은 `.claude/skills/`에 스킬로 축적하세요.
