# ARCTraj2JaxARC — Project Context

## Project Goal
ARCTraj 데이터셋(인간 ARC 풀이 궤적)을 JaxARC 환경 포맷으로 변환하여,
오프라인 RL/모방학습으로 ARC Task를 푸는 AI 모델을 학습하는 데 사용한다.

## Environment Setup
```bash
# Conda 환경 활성화
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate JaxARC
# JaxARC v1.0.2, Python 3.12
```

## Data Paths (원본 — 컴퓨터마다 다를 수 있음)
- ARCTraj CSV: `/home/sejin/IntentionLearning/dataset/ARCTraj/ARCTraj.csv` (10,669 trajectories)
- ARC-AGI-1 GT: `/mnt/c/Users/DSLAB/data/ARC-AGI-1/data/training/` (400 tasks)
- Output: `./output/arctraj_jaxarc.pkl` (7,501 converted trajectories)

## Key Technical Decisions

### 1. PRE-action grid semantics
ARCTraj의 `grid` 필드는 **액션 수행 전** 상태. 따라서:
- `state_t = action[t].grid`
- `next_state_t = action[t+1].grid` (다음 **operation**의 grid, selection 아님)

### 2. Selection-Operation 결합
ARCTraj는 SelectCell/SelectObject → Paint/Move 순서로 분리 기록.
연속된 Selection을 누적(OR)하여 다음 Operation의 selection mask로 결합.

### 3. State continuity 보장 (V3 fix)
O2ARC SelectCell이 grid를 수정함 → 바로 다음 action의 grid가 아닌,
**다음 Operation의 grid**를 next_state로 사용하여 `next_state[t] == state[t+1]` 보장.

### 4. Undo/Redo 제거
History 카테고리 액션을 단순 제거. 궤적 연속성은 위 next_state 로직으로 보장.

## Validation Results (V3 — 2025-02-20)
- Check 1 (Success): **7,501/7,501 PASS** (100%)
- Check 2 (Continuity): **50,629/50,629 PASS** (100%)
- Check 3 (No-op rate): 38.2% overall (대부분 ARCTraj 원본 데이터 특성)
- Check 4 (Selection validity): 100% within mask, 0 empty
- Check 5 (Value range): [0, 9] PASS
- Check 8 (Last=SUBMIT): 7,501/7,501 PASS

## File Structure
- `convert.py` — Main conversion pipeline (CSV → pkl)
- `mapping.py` — ARCTraj→JaxARC action mapping, selection building, trajectory conversion
- `dataset.py` — PyTorch Dataset + collate function
- `utils.py` — Grid padding, similarity computation
- `validate_v3.py` — Comprehensive validation script
- `check_move_noop.py` — Move no-op investigation
- `check_move_edge.py` — Move edge-clipping analysis
- `check_move_noop_reason.py` — Move no-op root cause analysis

## Next Steps
- [ ] 모델 아키텍처 선정 및 학습 파이프라인 구축
- [ ] No-op transition 필터링 옵션 추가 검토
- [ ] JaxARC 환경에서 변환된 궤적 replay 검증
