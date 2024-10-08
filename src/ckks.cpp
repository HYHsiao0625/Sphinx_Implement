#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>

#include "seal/seal.h"
#include "../include/ckks.hpp"

using namespace std;
using namespace seal;

// Initial parameters for ckks
void CKKS::initParams() {
    _param = EncryptionParameters(scheme_type::ckks);
    _param.set_poly_modulus_degree(8192);
    _param.set_coeff_modulus(CoeffModulus::Create(8192, {40, 30, 30, 30}));
}

// Function to Generate a Pair of Keys for Encrypt & Decrypt
void CKKS::generateKey(PublicKey* publickey, SecretKey* secretkey) {
    SEALContext _context(_param);
    KeyGenerator keygen(_context);
    keygen.create_public_key(*publickey);
    *secretkey = keygen.secret_key();
    keygen.create_relin_keys(_relinkeys);
}

void CKKS::Encoder(const double source, Plaintext* result) {
    SEALContext _context(_param);
    CKKSEncoder _codec(_context);
    _codec.encode(source, pow(2, 30), *result);
}

void CKKS::Decoder(const Plaintext source, double* result) {
    SEALContext _context(_param);
    CKKSEncoder _codec(_context);
    vector<double> _result;
    _codec.decode(source, _result);
    *result = (fabs(_result[0]) < 1e-6) ? 0.0 :  _result[0];
}

// Function to Encrypt Plaintext with Publickey
void CKKS::encryptPlain(const double plaintext, const PublicKey& publickey, Ciphertext* ciphertext) {
    SEALContext _context(_param);
    Plaintext encoded;
    Encoder(plaintext, &encoded);
    Encryptor encryptor(_context, publickey);
    encryptor.encrypt(encoded, *ciphertext);
}

// Function for Handling Operations on Ciphertext
void CKKS::evaluateCipher(Ciphertext* arg1, const char* specfied_operator, Ciphertext* arg2) {
    const char *operators[] = {"+", "-", "*", "^"};
    SEALContext _context(_param);
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
            evaluator.relinearize_inplace(*arg1, _relinkeys);
            break;
        }
        case 3: {
            evaluator.square_inplace(*arg1);
            evaluator.relinearize_inplace(*arg1, _relinkeys);
            break;
        }
        default: {
            cerr << "Invalid operator" << endl;
            break;
        }
    }
}

void CKKS::evaluatePlain(Ciphertext* arg1, const char* specfied_operator, const Plaintext* arg2) {
    const char *operators[] = {"+", "-", "*"};
    SEALContext _context(_param);
    Evaluator evaluator(_context);
    int index = -1;
    for (int i = 0; i < 3; i++) {
        if (!strcmp(specfied_operator, operators[i])) {
            index = i;
            break;
        }
    }

    switch (index) {
        case 0: {
            evaluator.add_plain_inplace(*arg1, *arg2);
            break;
        }
        case 1: {
            evaluator.sub_plain_inplace(*arg1, *arg2);
            break;
        }
        case 2: {
            evaluator.multiply_plain_inplace(*arg1, *arg2);
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
    Decryptor decryptor(_context, secretkey);
    Plaintext un_decode;
    decryptor.decrypt(ciphertext, un_decode);
    Decoder(un_decode, result);
}