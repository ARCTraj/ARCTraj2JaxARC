# ARCTraj2JaxARC — Project Context

## Project Goal
ARCTraj 데이터셋(인간 ARC 풀이 궤적)을 JaxARC 환경 포맷으로 변환하여,
오프라인 RL/모방학습으로 ARC Task를 푸는 AI 모델을 학습하는 데 사용한다.

## Environment Setup
```bash
# Conda 환경 활성화
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate JaxARC
# JaxARC v1.0.3, Python 3.12, JAX 0.9.0.1, PyTorch 2.9.0+cu126
```

## Data Paths (원본 — 컴퓨터마다 다를 수 있음)
- ARCTraj CSV: HuggingFace `SejinKimm/ARCTraj` → 로컬 다운로드 (10,670 trajectories)
- ARC-AGI-1 GT: `https://github.com/fchollet/ARC-AGI` → 로컬 clone (400 tasks)
- Output: `./output/arctraj_jaxarc.pkl` (10,193 converted trajectories)

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

### 5. Background color = -1 (JaxARC 호환)
JaxARC는 패딩 영역에 `-1`을 사용. 변환 시 30x30 패딩 영역을 `-1`로 채움.
(ARC 색상 0~9과 패딩을 구분하기 위함)

### 6. Trajectory Quality Flags
각 trajectory에 두 가지 플래그 부여:
- **`success`**: Submit 결과가 GT와 일치하는지 (bool)
- **`has_orphan_paste`**: Copy/CopyInput 없이 Paste를 수행한 비정상 trajectory (bool)

## Conversion Stats (success_only=False, 2026-02-23)
| 항목 | 수 |
|------|------|
| 원본 전체 | 10,670 |
| 변환 성공 | **10,193** |
| skipped_no_task | 476 (ARC-AGI-1 training 400개에 미포함) |
| skipped_error | 1 (빈 actionSequence) |

### Quality Breakdown
|  | Clean | Orphan Paste | Total |
|---|---:|---:|---:|
| **Success** | 7,110 | 391 | 7,501 |
| **Failed** | 2,574 | 118 | 2,692 |
| **Total** | **9,684** | **509** | **10,193** |

### 학습 시 필터링 예시
```python
# 모방학습용 (성공 + 정상만)
clean_success = [t for t in data if t["success"] and not t["has_orphan_paste"]]  # 7,110

# 오프라인 RL용 (실패 포함, 비정상만 제외)
all_clean = [t for t in data if not t["has_orphan_paste"]]  # 9,684
```

## Validation Results (success_only=True subset, 2026-02-23)
- Check 1 (Success): **7,501/7,501 PASS** (100%)
- Check 2 (Continuity): **50,629/50,629 PASS** (100%)
- Check 3 (No-op rate): 38.2% overall (대부분 ARCTraj 원본 데이터 특성)
- Check 4 (Selection validity): 100% within mask, 0 empty
- Check 5 (Value range): [-1, 9] PASS (패딩=-1, 색상=0~9)
- Check 8 (Last=SUBMIT): 7,501/7,501 PASS

## File Structure
- `convert.py` — Main conversion pipeline (CSV → pkl)
- `mapping.py` — ARCTraj→JaxARC action mapping, selection building, trajectory conversion
- `dataset.py` — PyTorch Dataset + collate function
- `utils.py` — Grid padding, similarity computation (background_color=-1)
- `validate_v3.py` — Comprehensive validation script
- `check_move_noop.py` — Move no-op investigation
- `check_move_edge.py` — Move edge-clipping analysis
- `check_move_noop_reason.py` — Move no-op root cause analysis

## Next Steps
- [ ] 모델 아키텍처 선정 및 학습 파이프라인 구축
- [ ] No-op transition 필터링 옵션 추가 검토
- [ ] JaxARC 환경에서 변환된 궤적 replay 검증
