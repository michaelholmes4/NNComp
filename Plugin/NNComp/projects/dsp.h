#pragma once
#include "Eigen/Dense"
#include <cmath>
#include "layers.h"
#include "gru-2-4.h"
#include "gru-4-2.h"
#include "gru-4-4.h"
#include "gru-8-1.h"
#include "gru-8-2.h"
#include "gru-8-4.h"
#include "gru-16-1.h"
#include "gru-16-2.h"
#include "gru-32-1.h"
#include "lstm-2-4.h"
#include "lstm-4-2.h"
#include "lstm-4-4.h"
#include "lstm-8-1.h"
#include "lstm-8-2.h"
#include "lstm-8-4.h"
#include "lstm-16-1.h"
#include "lstm-16-2.h"
#include "lstm-32-1.h"
#include "rnn-2-4.h"
#include "rnn-4-2.h"
#include "rnn-4-4.h"
#include "rnn-8-1.h"
#include "rnn-8-2.h"
#include "rnn-8-4.h"
#include "rnn-16-1.h"
#include "rnn-16-2.h"
#include "rnn-32-1.h"

/**
 * Class to apply nerual network to audio signal using Eigen matrix library
 */
template<typename T>
class NN
{
public:
  NN(){}
  
  /**
   * Process a single sample of audio using defined model
   */
  void ProcessSample(T* x, T* y, const int model)
  {
    switch(model) {
      case 0:
        m0.apply_model(x, y);
        break;
      case 1:
        m1.apply_model(x, y);
        break;
      case 2:
        m2.apply_model(x, y);
        break;
      case 3:
        m3.apply_model(x, y);
        break;
      case 4:
        m4.apply_model(x, y);
        break;
      case 5:
        m5.apply_model(x, y);
        break;
      case 6:
        m6.apply_model(x, y);
        break;
      case 7:
        m7.apply_model(x, y);
        break;
      case 8:
        m8.apply_model(x, y);
        break;
      case 9:
        m9.apply_model(x, y);
        break;
      case 10:
        m10.apply_model(x, y);
        break;
      case 11:
        m11.apply_model(x, y);
        break;
      case 12:
        m12.apply_model(x, y);
        break;
      case 13:
        m13.apply_model(x, y);
        break;
      case 14:
        m14.apply_model(x, y);
        break;
      case 15:
        m15.apply_model(x, y);
        break;
      case 16:
        m16.apply_model(x, y);
        break;
      case 17:
        m17.apply_model(x, y);
        break;
      case 18:
        m18.apply_model(x, y);
        break;
      case 19:
        m19.apply_model(x, y);
        break;
      case 20:
        m20.apply_model(x, y);
        break;
      case 21:
        m21.apply_model(x, y);
        break;
      case 22:
        m22.apply_model(x, y);
        break;
      case 23:
        m23.apply_model(x, y);
        break;
      case 24:
        m24.apply_model(x, y);
        break;
      case 25:
        m25.apply_model(x, y);
        break;
      case 26:
        m26.apply_model(x, y);
        break;
      default:
        //Pass output
        *y = *x;
    }
  }
  
  //Models
  Gru_32_1<T> m0;
  Gru_16_2<T> m1;
  Gru_8_4<T> m2;
  
  Gru_16_1<T> m3;
  Gru_8_2<T> m4;
  Gru_4_4<T> m5;
  
  Gru_8_1<T> m6;
  Gru_4_2<T> m7;
  Gru_2_4<T> m8;
  
  Lstm_32_1<T> m9;
  Lstm_16_2<T> m10;
  Lstm_8_4<T> m11;
  
  Lstm_16_1<T> m12;
  Lstm_8_2<T> m13;
  Lstm_4_4<T> m14;
  
  Lstm_8_1<T> m15;
  Lstm_4_2<T> m16;
  Lstm_2_4<T> m17;
  
  Rnn_32_1<T> m18;
  Rnn_16_2<T> m19;
  Rnn_8_4<T> m20;
  
  Rnn_16_1<T> m21;
  Rnn_8_2<T> m22;
  Rnn_4_4<T> m23;
  
  Rnn_8_1<T> m24;
  Rnn_4_2<T> m25;
  Rnn_2_4<T> m26;

};
