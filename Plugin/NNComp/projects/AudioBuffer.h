#pragma once
/*
 Very simple class to handle an audio buffer
 */
template<typename T, int nChans>
class CustomAudioBuffer
{
public:
  CustomAudioBuffer(){}
  
  void SetFrameLength(int length)
  {
    //Clear memory
    if(initialised != false)
    {
      for(int i = 0; i < nChans; i++)
      {
        delete [] buffer[i];
      }
      delete [] buffer;
    }
    
    //Create Buffer
    buffer = new T*[nChans];
    for(int i = 0; i < nChans; i++)
    {
      buffer[i] = new T[length];
    }
    initialised = true;
  }
  
  T** GetBuffer()
  {
    return buffer;
  }
  
private:
  T** buffer;
  bool initialised = false;
  
};
