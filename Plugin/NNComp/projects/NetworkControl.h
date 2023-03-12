#pragma once

#include "IControl.h"
#include "dsp.h"

BEGIN_IPLUG_NAMESPACE
BEGIN_IGRAPHICS_NAMESPACE

enum ModelType
{
  lstm = 0,
  gru,
  rnn
};

template<int MAX_HIDDEN, int MAX_LAYERS>
class NetworkControl : public ISwitchControlBase
{
public:
  NetworkControl(IRECT bounds, int paramIdx)
  : ISwitchControlBase(bounds, paramIdx, nullptr, 27)
  {
    CreateRectangles();
    SetParamIdx(paramIdx);
  }
  
  /**
   Main draw function
   */
  void Draw(IGraphics& g) override
  {
    //Draw circles
    IPattern gradient = IPattern::CreateLinearGradient(mDisplay.L, mDisplay.T, mDisplay.R, mDisplay.T);
    gradient.AddStop(IColor(255, 138, 248, 255), 0.0);
    gradient.AddStop(IColor(255, 107, 149, 255), 0.5);
    gradient.AddStop(IColor(255, 213, 124, 255), 1.0);
    
    if(!g.CheckLayer(mLayer))
    {
      g.StartLayer(this, mRECT);
      g.PathCircle(mCol[0], mDisplay.MH(), 7.); //Input
      for(int i = 0 ; i < nLayers; i ++) //Layers
      {
        for(int j = 0; j < nHidden; j++)
        {
          g.PathCircle(mCol[i + 1], mRow[j], 7.);
        }
      }
      g.PathCircle(mCol[nLayers + 1], mDisplay.MH(), 7.); //output
      g.PathStroke(gradient, 1.);
      
      //Draw Labels
      g.DrawText(tBody, "Input", IRECT(mCol[0] - 10., mDisplay.MH() + 20. , mCol[0] + 10., mDisplay.MH() + 35.));
      g.DrawText(tBody, "Output", IRECT(mCol[nLayers + 1] - 10., mDisplay.MH() + 20. , mCol[nLayers + 1] + 10., mDisplay.MH() + 35.));
      for(int i = 0 ; i < nLayers; i ++)
      {
        if(type == ModelType::gru)
        {
          g.DrawText(tBody, "GRU Layer", IRECT(mCol[i + 1] - 10., mRow[nHidden - 1] + 20., mCol[i + 1] + 10., mRow[nHidden - 1] + 35.));
        } else if(type == ModelType::lstm)
        {
          g.DrawText(tBody, "LSTM Layer", IRECT(mCol[i + 1] - 10., mRow[nHidden - 1] + 20., mCol[i + 1] + 10., mRow[nHidden - 1] + 35.));
        } else {
          g.DrawText(tBody, "RNN Layer", IRECT(mCol[i + 1] - 10., mRow[nHidden - 1] + 20., mCol[i + 1] + 10., mRow[nHidden - 1] + 35.));
        }
      }
      
      g.DrawText(tBody, "Linear Layer", IRECT(mCol[nLayers + 1] - 10., mRow[nHidden - 1] + 20., mCol[nLayers + 1] + 10., mRow[nHidden - 1] + 35.));
      g.PathClear();
      mLayer = g.EndLayer();
    }
    g.DrawLayer(mLayer);
    
    //Draw lines
    for(int j = 0; j < nHidden; j++)
    {
      g.DrawLine(COLOR_WHITE.WithOpacity(Clip((AmpToDB(weights[0][j]) + 60.)/100., 0.0, 1.0) + 0.05),mCol[0] + 12., mDisplay.MH(), mCol[1] - 12, mRow[j]);
    }
    for(int i = 0 ; i < nLayers - 1; i ++)
    {
      for(int j = 0; j < nHidden; j++)
      {
        for(int k = 0; k < nHidden; k++)
        {
          g.DrawLine(COLOR_WHITE.WithOpacity(Clip((AmpToDB(weights[i + 1][j + (k * nHidden)]) + 60.)/100., 0.0, 1.0) + 0.05), mCol[i + 1] + 12., mRow[j], mCol[i + 2] - 12., mRow[k]);
        }
      }
    }
    for(int j = 0; j < nHidden; j++)
    {
      g.DrawLine(COLOR_WHITE.WithOpacity(Clip((AmpToDB(weights[MAX_LAYERS][j]) + 60.)/100., 0.0, 1.0) + 0.05), mCol[nLayers] + 12., mRow[j], mCol[nLayers + 1] - 12, mDisplay.MH());
    }
  }
  
  void SetValue(double value, int valIdx) override
  {
    IControl::SetValue(value, valIdx);
    
    //Reset NN drawing layer
    SetupNetwork();
    
    if(mLayer != nullptr)
    {
      mLayer->Invalidate();
    }
  }
  
  void OnMsgFromDelegate(int msgTag, int dataSize, const void* pData) override
  {
    if (!IsDisabled() && msgTag == ISender<>::kUpdateMessage)
    {
      IByteStream stream(pData, dataSize);

      int pos = 0;
      ISenderData<MAX_LAYERS + 1, std::array<float, MAX_HIDDEN * MAX_HIDDEN>> d;
      pos = stream.Get(&d, pos);
      
      for(int l = 0; l < MAX_LAYERS + 1; l++)
      {
        for(int i = 0; i < MAX_HIDDEN * MAX_HIDDEN; i++)
        {
          weights[l][i] = d.vals[l][i];
        }
      }
      
      IControl::SetDirty(false);
    }
  }
  
protected:
  
  /**
   * Create all necessary rectangles
   */
  void CreateRectangles()
  {
    mDisplay = mRECT.GetPadded(-30.).GetVShifted(-10.);
    
    float width = mDisplay.W() / (nLayers + 2);
    float height = mDisplay.H() / nHidden;
    
    //Create column coordinates
    for(int i = 0; i < nLayers + 2; i++)
    {
      mCol[i] = mDisplay.L + (width * i) + (width/2);
    }
    
    //Create row coordinates
    for(int i = 0; i < nHidden; i++)
    {
      mRow[i] = mDisplay.T + height * i + (height / 2);
    }
  }
  
  void SetupNetwork()
  {
    int model = GetSelectedIdx();
    
    CreateRectangles();
    
    switch(model)
    {
      case 0:
        nHidden = 32;
        nLayers = 1;
        type = ModelType::gru;
        break;
      case 1:
        nHidden = 16;
        nLayers = 2;
        type = ModelType::gru;
        break;
      case 2:
        nHidden = 8;
        nLayers = 4;
        type = ModelType::gru;
        break;
      case 3:
        nHidden = 16;
        nLayers = 1;
        type = ModelType::gru;
        break;
      case 4:
        nHidden = 8;
        nLayers = 2;
        type = ModelType::gru;
        break;
      case 5:
        nHidden = 4;
        nLayers = 4;
        type = ModelType::gru;
        break;
      case 6:
        nHidden = 8;
        nLayers = 1;
        type = ModelType::gru;
        break;
      case 7:
        nHidden = 4;
        nLayers = 2;
        type = ModelType::gru;
        break;
      case 8:
        nHidden = 2;
        nLayers = 4;
        type = ModelType::gru;
        break;
      case 9:
        nHidden = 32;
        nLayers = 1;
        type = ModelType::lstm;
        break;
      case 10:
        nHidden = 16;
        nLayers = 2;
        type = ModelType::lstm;
        break;
      case 11:
        nHidden = 8;
        nLayers = 4;
        type = ModelType::lstm;
        break;
      case 12:
        nHidden = 16;
        nLayers = 1;
        type = ModelType::lstm;
        break;
      case 13:
        nHidden = 8;
        nLayers = 2;
        type = ModelType::lstm;
        break;
      case 14:
        nHidden = 4;
        nLayers = 4;
        type = ModelType::lstm;
        break;
      case 15:
        nHidden = 8;
        nLayers = 1;
        type = ModelType::lstm;
        break;
      case 16:
        nHidden = 4;
        nLayers = 2;
        type = ModelType::lstm;
        break;
      case 17:
        nHidden = 2;
        nLayers = 4;
        type = ModelType::lstm;
        break;
      case 18:
        nHidden = 32;
        nLayers = 1;
        type = ModelType::rnn;
        break;
      case 19:
        nHidden = 16;
        nLayers = 2;
        type = ModelType::rnn;
        break;
      case 20:
        nHidden = 8;
        nLayers = 4;
        type = ModelType::rnn;
        break;
      case 21:
        nHidden = 16;
        nLayers = 1;
        type = ModelType::rnn;
        break;
      case 22:
        nHidden = 8;
        nLayers = 2;
        type = ModelType::rnn;
        break;
      case 23:
        nHidden = 4;
        nLayers = 4;
        type = ModelType::rnn;
        break;
      case 24:
        nHidden = 8;
        nLayers = 1;
        type = ModelType::rnn;
        break;
      case 25:
        nHidden = 4;
        nLayers = 2;
        type = ModelType::rnn;
        break;
      case 26:
        nHidden = 2;
        nLayers = 4;
        type = ModelType::rnn;
        break;
    }
  }
  
private:
  int nHidden = 8;
  int nLayers = 4;
  ModelType type = ModelType::lstm;
  
  std::array<float, (MAX_LAYERS + 3)> mCol;
  std::array<float, MAX_HIDDEN> mRow;
  
  IRECT mDisplay;
  ILayerPtr mLayer;
  
  IColor cText = IColor(255,203,201,201);
  IText tBody = IText(12., cText, "Inter-Regular");
  
  std::array<float, MAX_HIDDEN * MAX_HIDDEN> weights[MAX_LAYERS + 1];

};

END_IGRAPHICS_NAMESPACE
END_IPLUG_NAMESPACE
