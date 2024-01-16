/*
 * hhhit-inl.h
 *
 *  Created on: Mar 28, 2014
 *      Author: meiermark
 */

#ifndef HHHIT_INL_H_
#define HHHIT_INL_H_

#include "util.h"

// /////////////////////////////////////////////////////////////////////////////////////
// //// Function for Viterbi()
// /////////////////////////////////////////////////////////////////////////////////////
// inline float max2(const float& xMM, const float& xX, char& b)
// {
//   if (xMM>xX) { b=MM; return xMM;} else { b=SAME;  return xX;}
// }
inline float max2(const float& xMM, const float& xSAME, char& b,
    const unsigned char bit) {
  if (xMM > xSAME) {
    b |= bit;
    return xMM;
  }
  else { /* b |= 0x00!*/
    return xSAME;
  }
}

/////////////////////////////////////////////////////////////////////////////////////
//// Functions that calculate P-values and probabilities
/////////////////////////////////////////////////////////////////////////////////////

//// Evaluate the CUMULATIVE extreme value distribution at point x
//// p(s)ds = lamda * exp{ -exp[-lamda*(s-mu)] - lamda*(s-mu) } ds = exp( -exp(-x) - x) dx = p(x) dx
//// => P(s>S) = integral_-inf^inf {p(x) dx}  = 1 - exp{ -exp[-lamda*(S-mu)] }
inline double Pvalue(double x, double a[]) {
  //a[0]=lamda, a[1]=mu
  double h = a[0] * (x - a[1]);
  return (h > 10) ? exp(-h) : double(1.0) - exp(-exp(-h));
}

inline double Pvalue(float x, float lamda, float mu) {
  double h = lamda * (x - mu);
  return (h > 10) ? exp(-h) : (double(1.0) - exp(-exp(-h)));
}

inline double logPvalue(float x, float lamda, float mu) {
  double h = lamda * (x - mu);
  return (h > 10) ? -h :
         (h < -2.5) ? -exp(-exp(-h)) : log((double(1.0) - exp(-exp(-h))));
}

inline double logPvalue(float x, double a[]) {
  double h = a[0] * (x - a[1]);
  return (h > 10) ? -h :
         (h < -2.5) ? -exp(-exp(-h)) : log((double(1.0) - exp(-exp(-h))));
}

const size_t F = sizeof(__m256) / sizeof(float);
const size_t QF = 4 * F;
inline float ExtractSum(__m256 a)
        {
            float __attribute__ ((aligned(32))) _a[8];
            _mm256_store_ps(_a, _mm256_hadd_ps(_mm256_hadd_ps(a, _mm256_setzero_ps()), _mm256_setzero_ps()));
            return _a[0] + _a[4];
        }

inline size_t AlignLo(size_t size, size_t align)
    {
        return size & ~(align - 1);
    }

 template <bool align> inline void NeuralProductSum(const float * a, const float * b, size_t offset, __m256 & sum)
        {
            __m256 _a = _mm256_loadu_ps(a + offset);
            __m256 _b = _mm256_loadu_ps(b + offset);
            sum = _mm256_fmadd_ps(_a, _b, sum);
        }

 template <bool align>  inline void NeuralProductSum(const float * a, const float * b, size_t size, float* sum)
        {

            *sum = 0;
            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            if (partialAlignedSize)
            {
                __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += QF)
                    {
                        NeuralProductSum<align>(a, b, i + F * 0, sums[0]);
                        NeuralProductSum<align>(a, b, i + F * 1, sums[1]);
                        NeuralProductSum<align>(a, b, i + F * 2, sums[2]);
                        NeuralProductSum<align>(a, b, i + F * 3, sums[3]);
                    }
                    sums[0] = _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3]));
                }
                for (; i < partialAlignedSize; i += F)
                    NeuralProductSum<align>(a, b, i, sums[0]);
                *sum += ExtractSum(sums[0]);
            }
            for (; i < size; ++i)
                *sum += a[i] * b[i];
        }

      inline double reduce_vector2(__m256d input) {
        __m256d temp = _mm256_hadd_pd(input, input);
        __m128d sum_high = _mm256_extractf128_pd(temp, 1);
        __m128d result = _mm_add_pd(sum_high, _mm256_castpd256_pd128(temp));
        return ((double*)&result)[0];
      }

      inline double dot_product(const double *a, const double *b, size_t N) {
        __m256d sum_vec = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);

        /* Add up partial dot-products in blocks of 256 bits */
        for(size_t ii = 0; ii < N/4; ++ii) {
          __m256d x = _mm256_loadu_pd(a+4*ii);
          __m256d y = _mm256_loadu_pd(b+4*ii);
          __m256d z = _mm256_mul_pd(x,y);
          sum_vec = _mm256_add_pd(sum_vec, z);
        }

        /* Find the partial dot-product for the remaining elements after
        * dealing with all 256-bit blocks. */
        double final = 0.0;
        for(size_t ii = N-N%4; ii < N; ++ii)
          final += a[ii] * b[ii];

        return reduce_vector2(sum_vec) + final;
      }
inline float ScalarProd20(const float* qi, const float* tj) {
    
//#ifdef AVX
//  float __attribute__((aligned(ALIGN_FLOAT))) res;
//  __m256 P; // query 128bit SSE2 register holding 4 floats
//  __m256 S; // aux register
//  __m256 R; // result
//  __m256* Qi = (__m256*) qi;
//  __m256* Tj = (__m256*) tj;
//
//  R = _mm256_mul_ps(*(Qi++),*(Tj++));
//  P = _mm256_mul_ps(*(Qi++),*(Tj++));
//  S = _mm256_mul_ps(*Qi,*Tj); // floats A, B, C, D, ?, ?, ? ,?
//  R = _mm256_add_ps(R,P);     // floats 0, 1, 2, 3, 4, 5, 6, 7
//  P = _mm256_permute2f128_ps(R, R, 0x01); // swap hi and lo 128 bits: 4, 5, 6, 7, 0, 1, 2, 3
//  R = _mm256_add_ps(R,P);     // 0+4, 1+5, 2+6, 3+7, 0+4, 1+5, 2+6, 3+7
//  R = _mm256_add_ps(R,S);     // 0+4+A, 1+5+B, 2+6+C, 3+7+D, ?, ?, ? ,?
//  R = _mm256_hadd_ps(R,R);    // 04A15B, 26C37D, ?, ?, 04A15B, 26C37D, ?, ?
//  R = _mm256_hadd_ps(R,R);    // 01234567ABCD, ?, 01234567ABCD, ?, 01234567ABCD, ?, 01234567ABCD, ?
//  _mm256_store_ps(&res, R);
//  return res;
//#else
#if 1
    float __attribute__((aligned(32))) res;

    __m256 zero256 = _mm256_setzero_ps();
    __m256 Qi0 = _mm256_loadu_ps(qi);
    __m256 Tj0 = _mm256_loadu_ps(tj);
    __m256 P0 = _mm256_mul_ps(Qi0, Tj0);
    __m256 Qi1 = _mm256_loadu_ps(qi + 8);
    __m256 Tj1 = _mm256_loadu_ps(tj + 8);
    __m256 P1 = _mm256_fmadd_ps(Qi1, Tj1, P0);
    __m256 Qi2 = _mm256_mask_loadu_ps(zero256, 0xf, qi + 16);
    __m256 Tj2 = _mm256_mask_loadu_ps(zero256, 0xf, tj + 16);
    __m256 P2 = _mm256_fmadd_ps(Qi2, Tj2, P1);
    __m256 P = _mm256_hadd_ps(P2, zero256);
    P = _mm256_hadd_ps(P, zero256);
    float  __attribute__((aligned(32))) temp[8];
    _mm256_store_ps(temp, P);
    res = temp[0] + temp[4];

    return res;
#else
#ifdef SSE
    float __attribute__((aligned(16))) res;
    __m128 P; // query 128bit SSE2 register holding 4 floats
    __m128 R;// result
    __m128* Qi = (__m128*) qi;
    __m128* Tj = (__m128*) tj;

    __m128 P1 = _mm_mul_ps(*(Qi),*(Tj));
    __m128 P2 = _mm_mul_ps(*(Qi+1),*(Tj+1));
    __m128 R1 = _mm_add_ps(P1, P2);

    __m128 P3 = _mm_mul_ps(*(Qi + 2), *(Tj + 2));
    __m128 P4 = _mm_mul_ps(*(Qi + 3), *(Tj + 3));
    __m128 R2 = _mm_add_ps(P3, P4);
    __m128 P5 = _mm_mul_ps(*(Qi+4), *(Tj+4));

    R = _mm_add_ps(R1, R2);
    R = _mm_add_ps(R,P5);

//    R = _mm_hadd_ps(R,R);
//    R = _mm_hadd_ps(R,R);
    P = _mm_shuffle_ps(R, R, _MM_SHUFFLE(2,0,2,0));
    R = _mm_shuffle_ps(R, R, _MM_SHUFFLE(3,1,3,1));
    R = _mm_add_ps(R,P);
    P = _mm_shuffle_ps(R, R, _MM_SHUFFLE(2,0,2,0));
    R = _mm_shuffle_ps(R, R, _MM_SHUFFLE(3,1,3,1));
    R = _mm_add_ps(R,P);
    _mm_store_ss(&res, R);
    return res;
#endif
#endif
}

// Calculate score between columns i and j of two HMMs (query and template)
inline float ProbFwd(float* qi, float* tj) {
  return ScalarProd20(qi, tj); //
}

//Calculate score between columns i and j of two HMMs (query and template)
inline float Score(float* qi, float* tj) {
  return fast_log2(ScalarProd20(qi, tj));
}

//// Calculate score between columns i and j of two HMMs (query and template)
//inline float ProbFwd(float* qi, float* tj) {
//  return ScalarProd20(qi, tj); //
//}
//
////Calculate score between columns i and j of two HMMs (query and template)
//inline float Score(float* qi, float* tj) {
//  return fast_log2(ProbFwd(qi, tj));
//}

#endif /* HHHIT_INL_H_ */
