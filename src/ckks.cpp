#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include <seal/seal.h>

using namespace std;
using namespace seal;

class CKKS {
    private:
        EncryptionParameters _param;

    public:
        void initParams();
        void generateKey(PublicKey* publickey, SecretKey* secretkey);
        void encryptPlain(const double plaintext, const PublicKey& publickey, Ciphertext* ciphertext);
        void evaluateCipher(Ciphertext* arg1, const char* specfied_operator, Ciphertext* arg2);
        void decryptCipher(const Ciphertext& ciphertext, const SecretKey& secretkey, double* result);
};

// Initial parameters for ckks
void CKKS::initParams() {
    _param = EncryptionParameters(scheme_type::ckks);
    _param.set_poly_modulus_degree(8192);
    _param.set_coeff_modulus(CoeffModulus::Create(8192, {60, 40, 40, 60}));
}

// Function to Generate a Pair of Keys for Encrypt & Decrypt
void CKKS::generateKey(PublicKey* publickey, SecretKey* secretkey) {
    SEALContext _context(_param);
    KeyGenerator keygen(_context);
    keygen.create_public_key(*publickey);
    *secretkey = keygen.secret_key();
}

// Function to Encrypt Plaintext with Publickey
void CKKS::encryptPlain(const double plaintext, const PublicKey& publickey, Ciphertext* ciphertext) {
    SEALContext _context(_param);
    CKKSEncoder _codec(_context);
    Plaintext encoded;
    _codec.encode(static_cast<double>(plaintext), pow(2, 40), encoded);
    Encryptor encryptor(_context, publickey);
    encryptor.encrypt(encoded, *ciphertext);
}

// Function for Handling Operations on Ciphertext
void CKKS::evaluateCipher(Ciphertext* arg1, const char* specfied_operator, Ciphertext* arg2 = nullptr) {
    const char *operators[] = {"+", "-", "*", "^"};
    SEALContext _context(_param);
    CKKSEncoder _codec(_context);
    Evaluator evaluator(_context);
    int index = -1;
    for (int i = 0; i < 4; i++) {
        if (!strcmp(specfied_operator, operators[i])) {
            index = i;
            break;
        }
    }

    switch (index) {
        case 0: {
            evaluator.add_inplace(*arg1, *arg2);
            break;
        }
        case 1: {
            evaluator.sub_inplace(*arg1, *arg2);
            break;
        }
        case 2: {
            evaluator.multiply_inplace(*arg1, *arg2);
            break;
        }
        case 3: {
            evaluator.square_inplace(*arg1);
            break;
        }
        default: {
            cerr << "Invalid operator" << endl;
            break;
        }
    }
}

// Function for Decrypting Ciphertext with SecretKey
void CKKS::decryptCipher(const Ciphertext& ciphertext, const SecretKey& secretkey, double* result) {
    SEALContext _context(_param);
    CKKSEncoder _codec(_context);
    Decryptor decryptor(_context, secretkey);
    Plaintext decoded;
    decryptor.decrypt(ciphertext, decoded);
    
    vector<double> _result;
    _codec.decode(decoded, _result);

    *result = (fabs(_result[0]) < 1e-6) ? 0.0 :  _result[0];
}