#include <iostream>
#include <fstream>
#include <cstdio>
#include "seal/seal.h"
#include "../include/ckks.hpp"

using namespace std;
using namespace seal;

template <typename T>

void outfile(string fileName, T& source) {
    ofstream outfile(fileName, ios::binary);
    source.save(outfile);
    outfile.close();
}

int main() {
    PublicKey _publickey;
    SecretKey _secretkey;

    CKKS _ckks;
    
    cout << "Initial encryption engine... ";
    _ckks.initParams();
    cout << "done.\nGenerate single-use keyset... ";
    _ckks.generateKey(&_publickey, &_secretkey);
    cout << "done.\n";

    vector<vector<double> > num_array(32, vector<double>(32, 0));
    vector<vector<Ciphertext> > cipher_array(32, vector<Ciphertext>(32));

    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++)
            num_array[i][j] = i * 32 + j + 1;

    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++)
            _ckks.encryptPlain(num_array[i][j], _publickey, &cipher_array[i][j]);

    Ciphertext divide10;

    _ckks.encryptPlain(0.01, _publickey, &divide10);

    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++)
            _ckks.evaluateCipher(&cipher_array[i][j], "*", &divide10);

    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++)
            _ckks.decryptCipher(cipher_array[i][j], _secretkey, &num_array[i][j]);

    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++)
            printf("%5.2f ", num_array[i][j]);
        cout << endl;
    }

    return 0;
}