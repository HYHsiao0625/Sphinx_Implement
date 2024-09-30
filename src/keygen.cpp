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

    vector<string> _operand(2);
    string _operator;
    cout << "done.\nType 1st operand: ";
    cin >> _operand[0];
    cout << "Get: " << _operand[0] << "\nType operator (+, -, *, ^): ";
    cin >> _operator;
    cout << "Get: " << _operator << "\nType 2nd operand: ";
    cin >> _operand[1];

    vector<Ciphertext> _cipher(2);
    cout << "Get: " << _operand[1] << "\nStart encrypting plaintext... ";
    for (int i = 0; i < 2; i++)
        _ckks.encryptPlain(stod(_operand[i]), _publickey, &_cipher[i]);
    cout << "done.\nStart evaluating ciphertext... ";

    double _res, plain[2];

    for (int i = 0; i < 2; i++)
        _ckks.decryptCipher(_cipher[i], _secretkey, &plain[i]);
    
    cout << "Decrypted Plaintext 1st: " << plain[0] << ", 2nd: " << plain[1] << endl;
  

    _ckks.evaluateCipher(&_cipher[0], _operator.c_str(), &_cipher[1]);
    cout << "done.\nStart decrypting ciphertext... ";
    _ckks.decryptCipher(_cipher[0], _secretkey, &_res);

    cout << "done.\nThe result is: " << _res << " \n";

    return 0;
}