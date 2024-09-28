#ifndef CKKS_HPP
#define CKKS_HPP

using namespace std;
using namespace seal;

class CKKS {
    private:
        EncryptionParameters _param;
        SEALContext _context;
        CKKSEncoder _codec;

    public:
        CKKS(): _param(scheme_type::ckks), _context(_param), _codec(_context) {}
        void initParams();
        void generateKey(PublicKey* publickey, SecretKey* secretkey);
        void encryptPlain(const double plaintext, PublicKey& publickey, Ciphertext* ciphertext, double scale = pow(2, 20));
        void evaluateCipher(Ciphertext* arg1, const char* specfied_operator, Ciphertext* arg2 = nullptr);
        void decryptCipher(const Ciphertext& ciphertext, const SecretKey& secretkey, double* result);
};

#endif