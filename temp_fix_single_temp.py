# TEMPORARY FIX: Single Temperature Version
# Copy this code to network/transformer_cam.py if multi-head version is still slow

# In __init__ method, REPLACE the multi-head version with:
if cross:
    # Temporary: Use single temperature (like original) 
    self.temperature = nn.Parameter(torch.ones(1))
    print(f"🔧 TEMP FIX: Using single temperature (not multi-head)")

# In forward method, REPLACE the multi-head version with:
if self.cross:
    dp = -1 * dp
    # Temporary: Use original single temperature
    attn = (dp / self.temperature).softmax(dim=-1)
else:
    attn = dp.softmax(dim=-1)