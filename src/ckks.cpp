#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cstdio>
#include "seal/seal.h"

using namespace std;
using namespace seal;

class CKKS {
    private:
        EncryptionParameters _param;
        SEALContext _context;
        CKKSEncoder _codec;

    public:
        CKKS(): _param(scheme_type::ckks), _context(_param), _codec(_param) {
            _param.set_poly_modulus_degree(8192);
            _param.set_coeff_modulus(CoeffModulus::Create(8192, {60, 40, 40, 60}));
        }
        ~CKKS() {}
        void generateKey(PublicKey *publickey, SecretKey *secretkey);
        void encryptPlain(double plaintext, PublicKey publickey, Ciphertext* ciphertext);
        void evaluateCipher(Ciphertext* arg1, const char* specfied_operator, Ciphertext* arg2 = nullptr);
        void decryptCipher(Ciphertext ciphertext, SecretKey secretkey, double *result);
};

// Function for Generate a Pair of Keys for Encrypt & Decrypt
void CKKS::generateKey(PublicKey *publickey, SecretKey *secretkey) {
    // Generate publickey
    KeyGenerator keygen(_context);

    // PublicKey public_key;
    keygen.create_public_key(*publickey);

    // Generate secretkey
    *secretkey = keygen.secret_key();
}

// Function for Encrypt Plaintext with Publickey
void CKKS::encryptPlain(double plaintext, PublicKey publickey, Ciphertext* ciphertext) {
    // Encode plaintext
    Plaintext encoded;
    _codec.encode(plaintext, pow(2.0, 40), encoded);

    // Setup encryptor
    Encryptor encryptor(_context, publickey);

    // Encrypt and save into ciphertext
    encryptor.encrypt(encoded, *ciphertext);
}

// Function for Handle Required Operations for Ciphertext
void CKKS::evaluateCipher(Ciphertext* arg1, const char* specfied_operator, Ciphertext* arg2 = nullptr) { 
    // Get require operation
    string operators[] = {"+", "-", "*", "^"};  // addition, subtraction, multiplication, square
    // Setup evaluator
    Evaluator evaluator(_context);

    // Translate into index
    int index = 0;
    for (int i = 0; i < 4; i++) {
        if (specfied_operator == operators[i]) {
            index = i;
            break;
        }
    }

    // Switch case
    switch (index) {
        case 0:
            evaluator.add_inplace(*arg1, *arg2);
            break;
        case 1:
            evaluator.sub_inplace(*arg1, *arg2);
            break;
        case 2:
            evaluator.multiply_inplace(*arg1, *arg2);
            break;
        case 3:
            evaluator.square_inplace(*arg1);
            break;
    }
}

// Function for decrypt ciphertext with secretkey
void CKKS::decryptCipher (Ciphertext ciphertext, SecretKey secretkey, double *result) {
    // Decrypt
    Decryptor decryptor(_context, secretkey);
    Plaintext undecode;
    decryptor.decrypt(ciphertext, undecode);
    vector<double> _result;
    _codec.decode(undecode, _result);

    *result = _result[0];
}