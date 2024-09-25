#ifndef CKKS_HPP
#define CKKS_HPP

using namespace std;
using namespace seal;

// Function for Generate a Pair of Keys for Encrypt & Decrypt
class CKKS {
    private:
        EncryptionParameters _param;
        SEALContext _context;
        CKKSEncoder _codec;

    public:
        CKKS();
        ~CKKS();
        void generateKey(PublicKey* publickey, SecretKey* secretkey);
        void encryptPlain(double plaintext, PublicKey publickey, Ciphertext* ciphertext);
        void evaluateCipher(Ciphertext* arg1, const char* specfied_operator, Ciphertext* arg2 = nullptr);
        void decryptCipher(Ciphertext ciphertext, SecretKey secretkey, double* result);
};

#endif