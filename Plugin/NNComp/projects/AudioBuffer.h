#pragma once
/*
 Very simple class to handle an audio buffer
 */
template<typename T, int nChans>
class CustomAudioBuffer
{
public:
  CustomAudioBuffer()
  {
  }
  
  void SetFrameLength(int length)
  {
    //resize buffer
    buffer.resize(length * nChans);
    
    //Reset pointers
    ptrs.clear();
    for(int i = 0; i < nChans; i++)
    {
      ptrs.push_back(&buffer[i * nChans]);
    }
  }
  
  T** GetBuffer()
  {
    return &ptrs[0];
  }
  
private:
  std::vector<T*> ptrs;
  std::vector<T> buffer;
};
