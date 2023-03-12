#pragma once
#include "IControl.h"
#include "NetworkControl.h"

BEGIN_IPLUG_NAMESPACE
BEGIN_IGRAPHICS_NAMESPACE

template <int MAX_HIDDEN, int MAX_LAYERS, int QUEUE_SIZE = 64>
class WeightSender : public ISender<MAX_LAYERS + 2, QUEUE_SIZE, std::array<float, MAX_HIDDEN * MAX_HIDDEN>>
{
public:
  WeightSender()
  : ISender<MAX_LAYERS + 2, QUEUE_SIZE, std::array<float, MAX_HIDDEN * MAX_HIDDEN>>()
  {
    
  }
  
  void ProcessWeights(NN<sample> nn, int ctrlTag = kNoTag, int model = 0)
  {
    switch(model)
    {
      case 0:
        ProcessGru_1(nn.m0, ctrlTag);
        break;
      case 1:
        ProcessGru_2(nn.m1, ctrlTag);
        break;
      case 2:
        ProcessGru_4(nn.m2, ctrlTag);
        break;
      case 3:
        ProcessGru_1(nn.m3, ctrlTag);
        break;
      case 4:
        ProcessGru_2(nn.m4, ctrlTag);
        break;
      case 5:
        ProcessGru_4(nn.m5, ctrlTag);
        break;
      case 6:
        ProcessGru_1(nn.m6, ctrlTag);
        break;
      case 7:
        ProcessGru_2(nn.m7, ctrlTag);
        break;
      case 8:
        ProcessGru_4(nn.m8, ctrlTag);
        break;
      case 9:
        ProcessLstm_1(nn.m9, ctrlTag);
        break;
      case 10:
        ProcessLstm_2(nn.m10, ctrlTag);
        break;
      case 11:
        ProcessLstm_4(nn.m11, ctrlTag);
        break;
      case 12:
        ProcessLstm_1(nn.m12, ctrlTag);
        break;
      case 13:
        ProcessLstm_2(nn.m13, ctrlTag);
        break;
      case 14:
        ProcessLstm_4(nn.m14, ctrlTag);
        break;
      case 15:
        ProcessLstm_1(nn.m15, ctrlTag);
        break;
      case 16:
        ProcessLstm_2(nn.m16, ctrlTag);
        break;
      case 17:
        ProcessLstm_4(nn.m17, ctrlTag);
        break;
      case 18:
        ProcessRnn_1(nn.m18, ctrlTag);
        break;
      case 19:
        ProcessRnn_2(nn.m19, ctrlTag);
        break;
      case 20:
        ProcessRnn_4(nn.m20, ctrlTag);
        break;
      case 21:
        ProcessRnn_1(nn.m21, ctrlTag);
        break;
      case 22:
        ProcessRnn_2(nn.m22, ctrlTag);
        break;
      case 23:
        ProcessRnn_4(nn.m23, ctrlTag);
        break;
      case 24:
        ProcessRnn_1(nn.m24, ctrlTag);
        break;
      case 25:
        ProcessRnn_2(nn.m25, ctrlTag);
        break;
      case 26:
        ProcessRnn_4(nn.m26, ctrlTag);
        break;
    }
      
  }
  
  template<typename T>
  void ProcessGru_1(T model, int ctrlTag)
  {
    ISenderData<MAX_LAYERS + 2, std::array<float, MAX_HIDDEN * MAX_HIDDEN>> d {ctrlTag, model.h, 0};
    
    //process input layer
    for(int i = 0; i < model.h; i++)
    {
      d.vals[0][i] = model.l0.Wir.coeff(i, 0) * model.l0.xt.coeff(i, 0);
    }
    
    //layer 1
    for(int i = 0; i < model.h; i++)
    {
      for(int o = 0; o < model.h; o++)
      {
        d.vals[1][i + (o * model.h)] = model.l0.ht[i] * model.f.A.coeff(i, o);
      }
    }
    
    //FCC Layer
    for(int i = 0; i < model.h; i++)
    {
      d.vals[MAX_LAYERS + 1][i] = model.f.y;
    }
    
    //Send data
    ISender<MAX_LAYERS + 2, QUEUE_SIZE, std::array<float, MAX_HIDDEN * MAX_HIDDEN>>::PushData(d);
  }
  
  template<typename T>
  void ProcessGru_2(T model, int ctrlTag)
  {
    ISenderData<MAX_LAYERS + 2, std::array<float, MAX_HIDDEN * MAX_HIDDEN>> d {ctrlTag, model.h, 0};
    
    //process input layer
    for(int i = 0; i < model.h; i++)
    {
      d.vals[0][i] = model.l0.Wir.coeff(i, 0) * model.l0.xt.coeff(i, 0);
    }
    
    //layers
    for(int i = 0; i < model.h; i++)
    {
      for(int o = 0; o < model.h; o++)
      {
        d.vals[1][i + (o * model.h)] = model.l0.ht[i] * model.l1.Wir.coeff(i, o);
        d.vals[2][i + (o * model.h)] = model.l1.ht[i] * model.f.A.coeff(i, o);
      }
    }
    
    //FCC layer
    for(int i = 0; i < model.h; i++)
    {
      d.vals[MAX_LAYERS + 1][i] = model.f.y;
    }
    
    //Send data
    ISender<MAX_LAYERS + 2, QUEUE_SIZE, std::array<float, MAX_HIDDEN * MAX_HIDDEN>>::PushData(d);
  }
  
  template<typename T>
  void ProcessGru_4(T model, int ctrlTag)
  {
    ISenderData<MAX_LAYERS + 2, std::array<float, MAX_HIDDEN * MAX_HIDDEN>> d {ctrlTag, model.h, 0};
    
    //process input layer
    for(int i = 0; i < model.h; i++)
    {
      d.vals[0][i] = model.l0.Wir.coeff(i, 0) * model.l0.xt.coeff(i, 0);
    }
    
    //layers
    for(int i = 0; i < model.h; i++)
    {
      for(int o = 0; o < model.h; o++)
      {
        d.vals[1][i + (o * model.h)] = model.l0.ht[i] * model.l1.Wir.coeff(i, o);
        d.vals[2][i + (o * model.h)] = model.l1.ht[i] * model.l2.Wir.coeff(i, o);
        d.vals[3][i + (o * model.h)] = model.l2.ht[i] * model.l3.Wir.coeff(i, o);
        d.vals[4][i + (o * model.h)] = model.l3.ht[i] * model.f.A.coeff(i, o);
      }
    }
    
    //Output layer
    for(int i = 0; i < model.h; i++)
    {
      d.vals[MAX_LAYERS + 1][i] = model.f.y;
    }
    
    //Send data
    ISender<MAX_LAYERS + 2, QUEUE_SIZE, std::array<float, MAX_HIDDEN * MAX_HIDDEN>>::PushData(d);
  }
  
  template<typename T>
  void ProcessLstm_1(T model, int ctrlTag)
  {
    ISenderData<MAX_LAYERS + 2, std::array<float, MAX_HIDDEN * MAX_HIDDEN>> d {ctrlTag, model.h, 0};
    
    //process input layer
    for(int i = 0; i < model.h; i++)
    {
      d.vals[0][i] = model.l0.Wii.coeff(i, 0) * model.l0.xt.coeff(i, 0);
    }
    
    //layer 1
    for(int i = 0; i < model.h; i++)
    {
      for(int o = 0; o < model.h; o++)
      {
        d.vals[1][i + (o * model.h)] = model.l0.ht[i] * model.f.A.coeff(i, o);
      }
    }
    
    //FCC Layer
    for(int i = 0; i < model.h; i++)
    {
      d.vals[MAX_LAYERS + 1][i] = model.f.y;
    }
    
    //Send data
    ISender<MAX_LAYERS + 2, QUEUE_SIZE, std::array<float, MAX_HIDDEN * MAX_HIDDEN>>::PushData(d);
  }
  
  template<typename T>
  void ProcessLstm_2(T model, int ctrlTag)
  {
    ISenderData<MAX_LAYERS + 2, std::array<float, MAX_HIDDEN * MAX_HIDDEN>> d {ctrlTag, model.h, 0};
    
    //process input layer
    for(int i = 0; i < model.h; i++)
    {
      d.vals[0][i] = model.l0.Wii.coeff(i, 0) * model.l0.xt.coeff(i, 0);
    }
    
    //layers
    for(int i = 0; i < model.h; i++)
    {
      for(int o = 0; o < model.h; o++)
      {
        d.vals[1][i + (o * model.h)] = model.l0.ht[i] * model.l1.Wii.coeff(i, o);
        d.vals[2][i + (o * model.h)] = model.l1.ht[i] * model.f.A.coeff(i, o);
      }
    }
    
    //FCC layer
    for(int i = 0; i < model.h; i++)
    {
      d.vals[MAX_LAYERS + 1][i] = model.f.y;
    }
    
    //Send data
    ISender<MAX_LAYERS + 2, QUEUE_SIZE, std::array<float, MAX_HIDDEN * MAX_HIDDEN>>::PushData(d);
  }
  
  template<typename T>
  void ProcessLstm_4(T model, int ctrlTag)
  {
    ISenderData<MAX_LAYERS + 2, std::array<float, MAX_HIDDEN * MAX_HIDDEN>> d {ctrlTag, model.h, 0};
    
    //process input layer
    for(int i = 0; i < model.h; i++)
    {
      d.vals[0][i] = model.l0.Wii.coeff(i, 0) * model.l0.xt.coeff(i, 0);
    }
    
    //layers
    for(int i = 0; i < model.h; i++)
    {
      for(int o = 0; o < model.h; o++)
      {
        d.vals[1][i + (o * model.h)] = model.l0.ht[i] * model.l1.Wii.coeff(i, o);
        d.vals[2][i + (o * model.h)] = model.l1.ht[i] * model.l2.Wii.coeff(i, o);
        d.vals[3][i + (o * model.h)] = model.l2.ht[i] * model.l3.Wii.coeff(i, o);
        d.vals[4][i + (o * model.h)] = model.l3.ht[i] * model.f.A.coeff(i, o);
      }
    }
    
    //Output layer
    for(int i = 0; i < model.h; i++)
    {
      d.vals[MAX_LAYERS + 1][i] = model.f.y;
    }
    
    //Send data
    ISender<MAX_LAYERS + 2, QUEUE_SIZE, std::array<float, MAX_HIDDEN * MAX_HIDDEN>>::PushData(d);
  }
  
  template<typename T>
  void ProcessRnn_1(T model, int ctrlTag)
  {
    ISenderData<MAX_LAYERS + 2, std::array<float, MAX_HIDDEN * MAX_HIDDEN>> d {ctrlTag, model.h, 0};
    
    //process input layer
    for(int i = 0; i < model.h; i++)
    {
      d.vals[0][i] = model.l0.Wih.coeff(i, 0) * model.l0.xt.coeff(i, 0);
    }
    
    //layer 1
    for(int i = 0; i < model.h; i++)
    {
      for(int o = 0; o < model.h; o++)
      {
        d.vals[1][i + (o * model.h)] = model.l0.ht[i] * model.f.A.coeff(i, o);
      }
    }
    
    //FCC Layer
    for(int i = 0; i < model.h; i++)
    {
      d.vals[MAX_LAYERS + 1][i] = model.f.y;
    }
    
    //Send data
    ISender<MAX_LAYERS + 2, QUEUE_SIZE, std::array<float, MAX_HIDDEN * MAX_HIDDEN>>::PushData(d);
  }
  
  template<typename T>
  void ProcessRnn_2(T model, int ctrlTag)
  {
    ISenderData<MAX_LAYERS + 2, std::array<float, MAX_HIDDEN * MAX_HIDDEN>> d {ctrlTag, model.h, 0};
    
    //process input layer
    for(int i = 0; i < model.h; i++)
    {
      d.vals[0][i] = model.l0.Wih.coeff(i, 0) * model.l0.xt.coeff(i, 0);
    }
    
    //layers
    for(int i = 0; i < model.h; i++)
    {
      for(int o = 0; o < model.h; o++)
      {
        d.vals[1][i + (o * model.h)] = model.l0.ht[i] * model.l1.Wih.coeff(i, o);
        d.vals[2][i + (o * model.h)] = model.l1.ht[i] * model.f.A.coeff(i, o);
      }
    }
    
    //FCC layer
    for(int i = 0; i < model.h; i++)
    {
      d.vals[MAX_LAYERS + 1][i] = model.f.y;
    }
    
    //Send data
    ISender<MAX_LAYERS + 2, QUEUE_SIZE, std::array<float, MAX_HIDDEN * MAX_HIDDEN>>::PushData(d);
  }
  
  template<typename T>
  void ProcessRnn_4(T model, int ctrlTag)
  {
    ISenderData<MAX_LAYERS + 2, std::array<float, MAX_HIDDEN * MAX_HIDDEN>> d {ctrlTag, model.h, 0};
    
    //process input layer
    for(int i = 0; i < model.h; i++)
    {
      d.vals[0][i] = model.l0.Wih.coeff(i, 0) * model.l0.xt.coeff(i, 0);
    }
    
    //layers
    for(int i = 0; i < model.h; i++)
    {
      for(int o = 0; o < model.h; o++)
      {
        d.vals[1][i + (o * model.h)] = model.l0.ht[i] * model.l1.Wih.coeff(i, o);
        d.vals[2][i + (o * model.h)] = model.l1.ht[i] * model.l2.Wih.coeff(i, o);
        d.vals[3][i + (o * model.h)] = model.l2.ht[i] * model.l3.Wih.coeff(i, o);
        d.vals[4][i + (o * model.h)] = model.l3.ht[i] * model.f.A.coeff(i, o);
      }
    }
    
    //Output layer
    for(int i = 0; i < model.h; i++)
    {
      d.vals[MAX_LAYERS + 1][i] = model.f.y;
    }
    
    //Send data
    ISender<MAX_LAYERS + 2, QUEUE_SIZE, std::array<float, MAX_HIDDEN * MAX_HIDDEN>>::PushData(d);
  }
  
private:
  
};

END_IGRAPHICS_NAMESPACE
END_IPLUG_NAMESPACE
