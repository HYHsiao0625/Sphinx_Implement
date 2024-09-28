#include <iostream>
#include <fstream>
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
    PublicKey publickey;
    SecretKey secretkey;

    CKKS ckks;
    
    ckks.generateKey(&publickey, &secretkey);

    // Generate publickey.bin
    outfile("publickey.bin", publickey);
    cout << "publickey.bin generated." << endl;
    // Generate secretkey.bin
    outfile("secretkey.bin", secretkey);
    cout << "secretkey.bin generated." << endl;

    return 0;
}