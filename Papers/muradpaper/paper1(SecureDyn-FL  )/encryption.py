import torch
from phe import paillier

# ==========================================
# 1. KEY GENERATION (System Initialization)
# ==========================================
def setup_cryptographic_context():
    """
    Generates the Public and Private keys for the Paillier cryptosystem.
    In the paper, the CA manages these and distributes the public key.
    """
    print("   [Crypto] Generating Paillier Keypair (1024-bit)...")
    public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)
    return public_key, private_key

# ==========================================
# 2. CLIENT-SIDE ENCRYPTION
# ==========================================
def encrypt_layer(public_key, layer_tensor):
    """
    Flattens a PyTorch tensor and encrypts each floating point number.
    Returns a list of Paillier EncryptedNumber objects.
    """
    flat_weights = layer_tensor.view(-1).cpu().tolist()
    # Encrypting individual floats
    encrypted_vector = [public_key.encrypt(x) for x in flat_weights]
    return encrypted_vector

# ==========================================
# 3. SERVER-SIDE HOMOMORPHIC AGGREGATION
# ==========================================
def homomorphic_averaging(encrypted_vectors, num_clients):
    """
    The server adds the ciphertexts together. 
    It never sees the raw weights.
    """
    print("   [Server] Performing Homomorphic Addition on Encrypted Tensors...")
    
    # Initialize the sum with the first client's encrypted weights
    summed_vector = list(encrypted_vectors[0])
    
    # Add the remaining clients' weights homomorphically
    for enc_vec in encrypted_vectors[1:]:
        for i in range(len(summed_vector)):
            summed_vector[i] = summed_vector[i] + enc_vec[i]
            
    # In Paillier, you can multiply a ciphertext by a plaintext scalar
    avg_vector = [x * (1.0 / num_clients) for x in summed_vector]
    return avg_vector

# ==========================================
# 4. CLIENT-SIDE DECRYPTION
# ==========================================
def decrypt_layer(private_key, encrypted_vector, original_shape):
    """
    The client decrypts the server's payload using the private key
    and reshapes it back into a PyTorch tensor.
    """
    decrypted_list = [private_key.decrypt(x) for x in encrypted_vector]
    decrypted_tensor = torch.tensor(decrypted_list).view(original_shape)
    return decrypted_tensor

if __name__ == "__main__":
    # --- QUICK PROOF OF CONCEPT TEST ---
    public_key, private_key = setup_cryptographic_context()
    
    # Simulate two clients with a small dummy layer
    client_1_weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    client_2_weights = torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0])
    
    # Clients encrypt their data (using Public Key)
    enc_1 = encrypt_layer(public_key, client_1_weights)
    enc_2 = encrypt_layer(public_key, client_2_weights)
    
    # Server averages them (Server DOES NOT have the Private Key)
    enc_avg = homomorphic_averaging([enc_1, enc_2], num_clients=2)
    
    # Client downloads and decrypts the result (using Private Key)
    result = decrypt_layer(private_key, enc_avg, original_shape=(5,))
    
    print(f"Client 1 Input: {client_1_weights.tolist()}")
    print(f"Client 2 Input: {client_2_weights.tolist()}")
    # The average should be [2.0, 3.0, 4.0, 5.0, 6.0]
    print(f"Decrypted Server Output: {[round(x, 4) for x in result.tolist()]}")