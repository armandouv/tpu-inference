import sys

with open("tpu_inference/runner/tpu_runner.py", "r") as f:
    lines = f.readlines()

out = []
state = 0
# 0: normal
# 1: inside is_prefill removal
# 2: inside if is_prefill
# 3: inside else
for line in lines:
    if state == 0:
        if line.strip() == "is_prefill = hidden_states.shape[0] > 1":
            state = 1
            continue
        out.append(line)
    elif state == 1:
        if line.strip() == "if is_prefill:":
            state = 2
            continue
    elif state == 2:
        if line.strip() == "else:":
            state = 3
            continue
        else:
            # We are inside if is_prefill, so unindent by 4 spaces
            if line.startswith("                "):
                out.append(line[4:])
            elif line.strip() == "":
                out.append(line)
            else:
                out.append(line) # fallback
    elif state == 3:
        if line.strip() == "all_tokens = []":
            state = 0
            out.append(line)
            continue

with open("tpu_inference/runner/tpu_runner.py", "w") as f:
    f.writelines(out)

