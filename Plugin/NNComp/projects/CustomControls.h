#pragma once
#include "IControl.h"

BEGIN_IPLUG_NAMESPACE
BEGIN_IGRAPHICS_NAMESPACE

/**
 * Custom meter sender
 */
template <int MAXNC = 1, int QUEUE_SIZE = 64>
class MeterSender : public ISender<MAXNC, QUEUE_SIZE, std::pair<float, float>>
{
public:
  class EnvelopeFollower {
   public:
    void Set(float sample_rate, float attack_time, float release_time)
    {
      attack_time_ = attack_time;
      release_time_ = release_time;
      sample_rate_ = sample_rate;
      CalculateCoefficients();
    }
    
    void CalculateCoefficients(void)
    {
      attack_coeff_ = std::exp(-1.0f / (attack_time_ * sample_rate_));
      release_coeff_ = std::exp(-1.0f / (release_time_ * sample_rate_));
    }

    float ProcessSample(float input)
    {
      float abs_input = std::abs(input);
      if (abs_input > envelope_) {
        // Attack
        envelope_ = attack_coeff_ * (envelope_ - abs_input) + abs_input;
      } else {
        // Release
        envelope_ = release_coeff_ * (envelope_ - abs_input) + abs_input;
      }
      return envelope_;
    }

   private:
    float sample_rate_ = DEFAULT_SAMPLE_RATE;
    float attack_time_;
    float release_time_;
    float attack_coeff_;
    float release_coeff_;
    float envelope_ = 0.0f;
  };

  class PeakHolder
  {
  public:
    void Set(float sample_rate, float hold_time, float decay_time)
    {
      _hold_time = hold_time;
      _decay_time = decay_time;
      _sample_rate = sample_rate;
      _count = _sample_rate * _hold_time;
      CalculateCoefficients();
    }
    
    void CalculateCoefficients(void)
    {
      _decay_coeff = std::exp(-1.0f / (_decay_time * _sample_rate));
    }
    
    float ProcessSample(float x)
    {
      float input = std::abs(x);
      
      if(input > _prev_value)
      {
        _count = _sample_rate * _hold_time;
        _prev_value = input;
        return input;
      } else {
        
        if(_count != 0) //Holding
        {
          _count --;
          return _prev_value;
        } else {
          _prev_value *= _decay_coeff;
          return _prev_value;
        }
      }
      
      return input;
    }
    
  private:
    float _hold_time;
    float _sample_rate;
    int _count;
    float _prev_value = 0.0;
    float _peak_value;
    float _decay_time;
    float _decay_coeff;
  };
  
  MeterSender(float windowSizeMs = 5.0f, float attack = 0.1, float release = 0.001, float hold = 1)
  : ISender<MAXNC, QUEUE_SIZE, std::pair<float, float>>()
  , mAttack(attack)
  , mRelease(release)
  , mHold(hold)
  {
    Reset(DEFAULT_SAMPLE_RATE);
  }
  
  void Reset(double sampleRate)
  {
    for(int i = 0; i < MAXNC; i++)
    {
      env[i].Set(sampleRate, mAttack/mWindowSize, mRelease/mWindowSize);
      pk[i].Set(sampleRate, mHold/mWindowSize, mRelease/mWindowSize);
    }
    SetWindowSizeMs(mWindowSizeMs, sampleRate);
    std::fill(mLevels.begin(), mLevels.end(), 0.0f);
  }
  
  
  void SetWindowSizeMs(double timeMs, double sampleRate)
  {
    mWindowSizeMs = timeMs;
    mWindowSize = static_cast<int>(timeMs * 0.001 * sampleRate);
  }
  
  void ProcessBlock(sample** inputs, int nFrames, int ctrlTag = kNoTag, int nChans = MAXNC, int chanOffset = 0)
  {
    for (auto s = 0; s < nFrames; s++)
    {
      //If start of new subframe send current peak value
      if (mCount == 0)
      {
        ISenderData<MAXNC, std::pair<float, float>> d {ctrlTag, nChans, chanOffset};
        for (auto c = chanOffset; c < (chanOffset + nChans); c++)
        {
          //Apply ballistics
          mLevels[c] = env[c].ProcessSample(mLevels[c]);
          mPeaks[c] = pk[c].ProcessSample(mLevels[c]);
          
          std::get<0>(d.vals[c]) = mLevels[c]; //Set output current levels
          std::get<1>(d.vals[c]) = mPeaks[c]; //Set output current peaks
          mLevels[c] = 0.0f; //Reset current levels
          mPeaks[c] = 0.0f; //Reset current peaks
        }
        ISender<MAXNC, QUEUE_SIZE, std::pair<float, float>>::PushData(d);
      }
      
      //Find max value in buffer
      for (auto c = chanOffset; c < (chanOffset + nChans); c++)
      {
        mLevels[c] = (mLevels[c] > std::fabs(inputs[c][s])) ? mLevels[c] : std::fabs(inputs[c][s]);
      }
      
      mCount++;
      mCount %= mWindowSize;
    }
  }
private:
  float mWindowSizeMs = 5.0f;
  int mWindowSize = 32;
  int mCount = 0;
  std::array<float, MAXNC> mLevels = {0.0}; //Smoothed level
  std::array<float, MAXNC> mPeaks = {0.0}; //Peak hold
  float mAttack;
  float mRelease;
  float mHold;
  
  //Envelope followers
  std::array<EnvelopeFollower, MAXNC> env;
  
  //Peak holders
  std::array<PeakHolder, MAXNC> pk;
};

/**
 * Create the background of the plugin
 */
class BackgroundControl : public IControl
{
public:
  BackgroundControl(IRECT bounds) : IControl(bounds)
  {
    CreateRectangles();
  }
  
  void Draw(IGraphics& g) override
  {
    g.FillRect(cBg, mRECT);
    if (!g.CheckLayer(mLayer))
    {
      g.StartLayer(this, mRECT);
      g.FillRoundRect(cPanel, mTopPanel, 10.);
      g.FillRoundRect(cPanel, mLeftPanel, 10.);
      g.FillRoundRect(cPanel, mRightPanel, 10.);
      g.FillRoundRect(cPanel, mMainPanel, 10.);
      mLayer = g.EndLayer();
      g.ApplyLayerDropShadow(mLayer, mShadow);
    }
    g.DrawLayer(mLayer);
    
    if (!g.CheckLayer(mLayerFg))
    {
      g.StartLayer(this, mRECT);
      ISVG logo = g.LoadSVG(LOGO_SVG);
      g.DrawSVG(logo, mTopPanel.GetInsideSize(12.52, 10.51, 166.78, 37.67));
      g.DrawText(tBody, "Visualisations of 27 different nerual network combinations modelling a classic FET compressor", mTopPanel.GetInsideSize(365, 21, 446, 12));
      mLayerFg = g.EndLayer();
    }
    g.DrawLayer(mLayerFg);
    
  }
  
  void CreateRectangles()
  {
    mTopPanel = mRECT.GetInsideSize(7, 5, 822, 55);
    mLeftPanel = mRECT.GetInsideSize(7, 224, 70, 294);
    mRightPanel = mRECT.GetInsideSize(759, 224, 70, 294);
    mMainPanel = mRECT.GetInsideSize(87, 97, 662, 559);
  }
  
  //void OnResize() override;
  //void OnInit() override;

private:
  IRECT mTopPanel;
  IRECT mLeftPanel;
  IRECT mRightPanel;
  IRECT mMainPanel;
  IColor cBg = IColor(255, 2, 10, 32);
  IColor cPanel = IColor(25, 119, 119, 119);
  IColor cText = IColor(255,203,201,201);
  IShadow mShadow = IShadow(COLOR_BLACK, 4.0, 5.0, 5.0, 100.f, true);
  IText tBody = IText(12., cText, "Inter-Regular");
  ILayerPtr mLayer;
  ILayerPtr mLayerFg;
};

/**
 * Creates a nice looking gradient meter
 */
class MeterControl : public IControl
{
public:
  MeterControl(IRECT bounds)
  : IControl(bounds)
  , mBounds(bounds)
  {
    CreateAssets();
    mFmtString.Set("%.1f");
  }
  
  void Draw(IGraphics& g) override
  {
    //Draw background
    if (!g.CheckLayer(mLayerBg))
    {
      g.StartLayer(this, mRECT);
      IPattern gradient = IPattern::CreateLinearGradient(mLeftChannelArea.L, mLeftChannelArea.T, mLeftChannelArea.L, mLeftChannelArea.B);
      gradient.AddStop(IColor(25, 138, 248, 255), 0.0);
      gradient.AddStop(IColor(25, 107, 149, 255), 0.5);
      gradient.AddStop(IColor(25, 213, 124, 255), 1.0);
      g.PathRoundRect(mLeftChannelArea, 6.);
      g.PathRoundRect(mRightChannelArea, 6.);
      g.PathFill(gradient);
      mLayerBg = g.EndLayer();
    }
    g.DrawLayer(mLayerBg);
    
    //Draw Tracks
    IPattern gradient = IPattern::CreateLinearGradient(mLeftChannelArea.L, mLeftChannelArea.T, mLeftChannelArea.L, mLeftChannelArea.B);
    gradient.AddStop(IColor(255, 138, 248, 255), 0.0);
    gradient.AddStop(IColor(255, 107, 149, 255), 0.5);
    gradient.AddStop(IColor(255, 213, 124, 255), 1.0);
    if(mValues[0] > 0.01)
    {
      if(mValues[0] < 1.0)
      {
        g.PathRoundRect(mLeftChannelArea.GetFromBottom(mValues[0] * mLeftChannelArea.H()), 6.);
      } else
      {
        g.PathRoundRect(mLeftChannelArea.GetFromBottom(mLeftChannelArea.H()), 6.);
      }
    }
    if(mValues[1] > 0.01)
    {
      if(mValues[0] < 1.0)
      {
        g.PathRoundRect(mRightChannelArea.GetFromBottom(mValues[1] * mRightChannelArea.H()), 6.);
      } else
      {
        g.PathRoundRect(mRightChannelArea.GetFromBottom(mRightChannelArea.H()), 6.);
      }
    }
    g.PathFill(gradient);
    
    //Draw Values
    g.DrawText(tValue, mPeakValues[0].Get(), mValuesArea.GetFromLeft(mValuesArea.W()/2).GetFromRight(30.));
    g.DrawText(tValue, mPeakValues[1].Get(), mValuesArea.GetFromRight(mValuesArea.W()/2).GetFromLeft(30.));
    
  }
  
  void OnMsgFromDelegate(int msgTag, int dataSize, const void* pData) override
  {
    if (!IsDisabled() && msgTag == ISender<>::kUpdateMessage)
    {
      IByteStream stream(pData, dataSize);

      int pos = 0;
      ISenderData<2, std::pair<float, float>> d;
      pos = stream.Get(&d, pos);
      
      const auto lowRangeDB = -60.;
      const auto highRangeDB = 0.;

      double lowPointAbs = std::fabs(lowRangeDB);
      double rangeDB = std::fabs(highRangeDB - lowRangeDB);
      
      for (auto c = d.chanOffset; c < (d.chanOffset + d.nChans); c++)
      {
        double value = AmpToDB(static_cast<double>(std::get<0>(d.vals[c])));
        double linearPos = (value + lowPointAbs)/rangeDB;
        
        mValues[c] = static_cast<float>(linearPos);
        
        double peakValue = AmpToDB(static_cast<double>(std::get<1>(d.vals[c])));
        if(peakValue > lowRangeDB)
        {
          mPeakValues[c].SetFormatted(256, mFmtString.Get(), peakValue);
        } else {
          mPeakValues[c].Set("-inf");
        }
      }
      
      IControl::SetDirty(false);
    }
  }
  
  void CreateAssets()
  {
    mValuesArea = mRECT.GetFromTop(12.);
    mLeftChannelArea = mRECT.GetReducedFromTop(12. + 4.).GetFromLeft(mRECT.W()/2.).GetFromRight(15.).GetHShifted(-3);
    mRightChannelArea = mRECT.GetReducedFromTop(12. + 4.).GetFromRight(mRECT.W()/2.).GetFromLeft(15.).GetHShifted(3);
  }
  
private:
  IRECT mBounds;
  IRECT mLeftChannelArea;
  IRECT mRightChannelArea;
  IRECT mValuesArea;
  ILayerPtr mLayerBg;
  IPattern gradient = IPattern::CreateLinearGradient(0.,0.,0.,0);
  
  std::array<float, 2> mValues;
  std::array<WDL_String, 2> mPeakValues;
  WDL_String mFmtString;
  
  IColor cText = IColor(255,156,156,156);
  IText tValue = IText(12., cText, "Inter-Regular");
};

class KnobControl : public IVKnobControl
{
public:
  KnobControl(IRECT bounds, int paramIdx, const char* label)
  : IVKnobControl(bounds.GetPadded(20.), paramIdx, label, DEFAULT_STYLE.WithValueText(IText(12., IColor(255, 156, 156, 156), "Inter-Regular")).WithLabelText(IText(12., IColor(255, 208, 208, 208), "Inter-Regular")), true, false, -135., 135., -135., EDirection::Vertical, 8.)
  {
    
  }
  
  void OnResize() override
  {
    mWidgetBounds = mRECT.GetCentredInside(34.);
    mLabelBounds = mRECT.GetFromBottom(14.).GetVShifted(-20.);
    mValueBounds = mRECT.GetFromTop(14.).GetVShifted(20.).GetCentredInside(IRECT(0,0,30.,14.));

    SetTargetRECT(mWidgetBounds);
    SetDirty(false);
  }
  
  void DrawWidget(IGraphics& g) override
  {
    //Draw fill
    if (!g.CheckLayer(mLayerBg))
    {
      g.StartLayer(this, mRECT);
      g.PathCircle(mWidgetBounds.L + mWidgetBounds.W()/2, mWidgetBounds.T + mWidgetBounds.W()/2, mWidgetBounds.W()/2);
      IPattern glow = IPattern::CreateRadialGradient(mWidgetBounds.L + mWidgetBounds.W()/2, mWidgetBounds.T + mWidgetBounds.W()/2, mWidgetBounds.W()/2);
      glow.AddStop(COLOR_BLACK, 0.0);
      if(!mMouseIsOver)
      {
        glow.AddStop(IColor(255,47,54,82), 1.0);
      } else
      {
        glow.AddStop(IColor(255,67,74,102), 1.0);
      }
      g.PathFill(glow);
      mLayerBg = g.EndLayer();
      g.ApplyLayerDropShadow(mLayerBg, IShadow(COLOR_BLACK, 4., 4., 4., 0.25));
    }
    g.DrawLayer(mLayerBg);
    
    if (!g.CheckLayer(mLayerRing))
    {
      g.StartLayer(this, mRECT);
      IPattern grad = IPattern::CreateLinearGradient(mWidgetBounds.L, mWidgetBounds.T, mWidgetBounds.R, mWidgetBounds.T);
      grad.AddStop(IColor(255, 138, 248, 255), 0.0);
      grad.AddStop(IColor(255, 107, 149, 255), 0.5);
      grad.AddStop(IColor(255, 213, 124, 255), 1.0);
      g.PathCircle(mWidgetBounds.L + mWidgetBounds.W()/2, mWidgetBounds.T + mWidgetBounds.W()/2, mWidgetBounds.W()/2);
      g.PathStroke(grad, 2.0);
      mLayerRing = g.EndLayer();
    }
    g.DrawLayer(mLayerRing);
    
    const float angle = mAngle1 + (static_cast<float>(GetValue()) * (mAngle2 - mAngle1));
    
    g.DrawRadialLine(IColor(255,208,208,208), mWidgetBounds.L + mWidgetBounds.W()/2, mWidgetBounds.T + mWidgetBounds.W()/2, angle, 10., 18., 0 , 2.);
  }
  
  void OnMouseOver(float x, float y, const IMouseMod& mod) override
  {
    bool prev = mMouseIsOver;
    mMouseIsOver = true;
    if (prev == false)
      SetDirty(false);
      mLayerBg->Invalidate();
  }
  
  void OnMouseOut() override
  {
    bool prev = mMouseIsOver;
    mMouseIsOver = false;
    if (prev == true)
      SetDirty(false);
      mLayerBg->Invalidate();
  }
  
private:
  ILayerPtr mLayerBg;
  ILayerPtr mLayerRing;
};


class DropDownControl : public ICaptionControl
{
public:
  DropDownControl(const IRECT& bounds, int paramIdx)
  : ICaptionControl(bounds.GetPadded(20.), paramIdx, IText(12., COLOR_WHITE.WithOpacity(0.4), "Inter-Regular"))
  {
    mButton = mRECT.GetPadded(-20.);
  }
  
  void Draw(IGraphics& g) override
  {
    const IParam* pParam = GetParam();
    if(pParam)
    {
      pParam->GetDisplay(mStr);
    }
    
    if(!g.CheckLayer(mLayer))
    {
      g.StartLayer(this, mRECT);
      IPattern grad = IPattern::CreateLinearGradient(mButton.L, mButton.T, mButton.R, mButton.T);
      if(!mMouseIsOver)
      {
        grad.AddStop(IColor(51, 138, 248, 255), 0.0);
        grad.AddStop(IColor(51, 107, 149, 255), 0.5);
        grad.AddStop(IColor(51, 213, 124, 255), 1.0);
      } else {
        grad.AddStop(IColor(100, 138, 248, 255), 0.0);
        grad.AddStop(IColor(100, 107, 149, 255), 0.5);
        grad.AddStop(IColor(100, 213, 124, 255), 1.0);
      }
      
      g.PathRoundRect(mButton, 6.);
      g.PathFill(grad);
      mLayer = g.EndLayer();
      g.ApplyLayerDropShadow(mLayer, mShadow);
    }
    g.DrawLayer(mLayer);
    IRECT tri = mButton.GetFromLeft(mButton.H()).GetCentredInside(6).GetHShifted(3.);
    g.FillTriangle(COLOR_WHITE.WithOpacity(0.2), tri.L, tri.T, tri.R, tri.T, tri.MW(), tri.B);
    
    ITextControl::Draw(g);
  }
  
  void OnMouseDown(float x, float y, const IMouseMod& mod) override
  {
    if (mod.L || mod.R)
    {
      if(mButton.Contains(x, y))
      {
        PromptUserInput(mButton);
      }
    }
  }
  
  void OnMouseOver(float x, float y, const IMouseMod& mod) override
  {
    if(mButton.Contains(x, y))
    {
      bool prev = mMouseIsOver;
      mMouseIsOver = true;
      if (prev == false)
        SetDirty(false);
        mLayer->Invalidate();
    }
  }
  
  void OnMouseOut() override
  {
    bool prev = mMouseIsOver;
    mMouseIsOver = false;
    if (prev == true)
      SetDirty(false);
      mLayer->Invalidate();
  }
  
private:
  IRECT mButton;
  ILayerPtr mLayer;
  IShadow mShadow = IShadow(COLOR_BLACK, 4.0, 4.0, 4.0, 100.f, true);
};

class GrMeterControl : public IControl
{
public:
  GrMeterControl(IRECT bounds)
  : IControl(bounds)
  , mBounds(bounds)
  {
    CreateAssets();
    mFmtString.Set("%.1f");
  }
  
  void Draw(IGraphics& g) override
  {
    //Draw background
    if (!g.CheckLayer(mLayerBg))
    {
      g.StartLayer(this, mRECT);
      IPattern gradient = IPattern::CreateLinearGradient(mChannelArea.R, mChannelArea.T, mChannelArea.L, mChannelArea.T);
      gradient.AddStop(IColor(25, 138, 248, 255), 0.0);
      gradient.AddStop(IColor(25, 107, 149, 255), 0.5);
      gradient.AddStop(IColor(25, 213, 124, 255), 1.0);
      g.PathRoundRect(mChannelArea, 6.);
      g.PathFill(gradient);
      mLayerBg = g.EndLayer();
    }
    g.DrawLayer(mLayerBg);
    
    //Draw Tracks
    IPattern gradient = IPattern::CreateLinearGradient(mChannelArea.R, mChannelArea.T, mChannelArea.L, mChannelArea.R);
    gradient.AddStop(IColor(255, 138, 248, 255), 0.0);
    gradient.AddStop(IColor(255, 107, 149, 255), 0.5);
    gradient.AddStop(IColor(255, 213, 124, 255), 1.0);
    if(mValues[0] > 0.01)
    {
      if(mValues[0] < 1.0)
      {
        g.PathRoundRect(mChannelArea.GetFromRight(mValues[0] * mChannelArea.W()), 6.);
      } else
      {
        g.PathRoundRect(mChannelArea.GetFromRight(mChannelArea.W()), 6.);
      }
    }
    g.PathFill(gradient);
    
    //Draw Values
    g.DrawText(tValue, mPeakValues[0].Get(), mValuesArea);
  }
  
  void OnMsgFromDelegate(int msgTag, int dataSize, const void* pData) override
  {
    if (!IsDisabled() && msgTag == ISender<>::kUpdateMessage)
    {
      IByteStream stream(pData, dataSize);

      int pos = 0;
      ISenderData<2, std::pair<float, float>> d;
      pos = stream.Get(&d, pos);
      
      const auto lowRangeDB = -40.;
      const auto highRangeDB = -20.;

      double lowPointAbs = std::fabs(lowRangeDB);
      double rangeDB = std::fabs(highRangeDB - lowRangeDB);
      
      for (auto c = d.chanOffset; c < (d.chanOffset + d.nChans); c++)
      {
        double value = AmpToDB(static_cast<double>(std::get<0>(d.vals[c])));
        double linearPos = (value + lowPointAbs)/rangeDB;
        
        mValues[c] = static_cast<float>(linearPos);
        
        double peakValue = AmpToDB(static_cast<double>(std::get<1>(d.vals[c])));
        if(peakValue > lowRangeDB)
        {
          mPeakValues[c].SetFormatted(256, mFmtString.Get(), lowRangeDB - peakValue);
        } else {
          mPeakValues[c].Set("0.0");
        }
      }
      
      IControl::SetDirty(false);
    }
  }
  
  void CreateAssets()
  {
    mValuesArea = mRECT.GetFromRight(30.);
    mChannelArea = mRECT.GetFromLeft(170.);
  }
  
private:
  IRECT mBounds;
  IRECT mChannelArea;
  IRECT mValuesArea;
  ILayerPtr mLayerBg;
  IPattern gradient = IPattern::CreateLinearGradient(0.,0.,0.,0);
  
  std::array<float, 2> mValues;
  std::array<WDL_String, 2> mPeakValues;
  WDL_String mFmtString;
  
  IColor cText = IColor(255,156,156,156);
  IText tValue = IText(12., cText, "Inter-Regular");
};

END_IGRAPHICS_NAMESPACE
END_IPLUG_NAMESPACE
