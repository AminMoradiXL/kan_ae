grid = 5
input_size = 28 * 28 # 28 *28, 32 * 32 * 3
hidden_size = 8 # 8, 32, 16
bottleneck_size = 9 # 

n_enc = grid * input_size * hidden_size
n_enc_dec = hidden_size * (bottleneck_size + 1)
n_dec_enc = bottleneck_size * (hidden_size + 1) 
n_dec = grid * input_size * hidden_size


n_param_kan_real = n_enc + n_enc_dec + n_dec_enc + n_dec 

n_param_kan = grid * input_size * hidden_size + hidden_size * (bottleneck_size + 1) + grid * bottleneck_size * hidden_size + bottleneck_size * (hidden_size + 1) 

print(f'number of parameters AE-KAN = {n_param_kan}')