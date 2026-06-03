// Benchmark tool: convert .mtx (COO) coordinates into various target formats
// Measures pack time and storage memory

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>

#include "taco.h"
#include "taco/storage/pack.h"
#include "taco/storage/typed_vector.h"
#include "taco/util/timers.h"
#include "taco/error.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <cstring>

using namespace std;
using namespace taco;

static bool parseMTX(const string &filename, vector<int> &dims,
                     vector<vector<int>> &coords, vector<double> &values) {
  ifstream file(filename);
  if (!file.is_open()) return false;

  string line;
  // read header line
  if (!getline(file, line)) return false;

  // skip comments until dimension line
  while (getline(file, line)) {
    if (line.size() == 0) continue;
    if (line[0] == '%') continue;
    stringstream ss(line);
    vector<int> header;
    int v;
    while (ss >> v) header.push_back(v);
    if (header.size() < 3) return false;
    int nnz = header.back(); header.pop_back();
    dims = header;
    coords.assign(dims.size(), vector<int>());
    values.reserve(nnz);
    // read nnz lines
    for (int i=0;i<nnz;i++){
      if (!getline(file, line)) break;
      if (line.size() == 0) { i--; continue; }
      if (line[0] == '%') { i--; continue; }
      stringstream ls(line);
      for (size_t m=0;m<dims.size();m++){ int idx; ls >> idx; coords[m].push_back(idx-1); }
      double val; ls >> val; values.push_back(val);
    }
    break;
  }

  file.close();
  return true;
}

static size_t getVmRSSKB() {
  ifstream stat("/proc/self/status");
  if (!stat.is_open()) return 0;
  string line;
  while (getline(stat, line)){
    if (line.rfind("VmRSS:", 0) == 0) {
      stringstream ss(line);
      string key; size_t kb; string unit;
      ss >> key >> kb >> unit; return kb;
    }
  }
  return 0;
}

Format makeFormat(const string &name, int order) {
  if (name=="csr") return Format({Dense, Compressed});
  if (name=="csc") return Format({Compressed, Dense});
  if (name=="coo") return COO(order);
  // fallback: compressed for all modes
  vector<ModeFormatPack> packs; packs.reserve(order);
  for (int i=0;i<order;i++) packs.push_back(ModeFormatPack({Sparse}));
  return Format(packs);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: benchmark [--repeat N] [--cold] [--formats f1,f2,...] file1.mtx [file2.mtx ...]" << endl;
    return 1;
  }

  int repeat = 5; bool cold=false; vector<string> formats = {"csc","ell"};
  string outDir = "data/benchmarks";
  vector<string> files;
  for (int i=1;i<argc;i++){
    string s = argv[i];
    if (s=="--repeat" && i+1<argc) { repeat = stoi(argv[++i]); continue; }
    if (s=="--cold") { cold=true; continue; }
    if (s=="--formats" && i+1<argc){ formats.clear(); string f=argv[++i]; string tmp; stringstream ss(f); while(getline(ss,tmp,',')) formats.push_back(tmp); continue; }
    if (s=="--out-dir" && i+1<argc) { outDir = argv[++i]; continue; }
    files.push_back(s);
  }

  cout << "matrix,format,mean_ms,stdev_ms,median_ms,samples,vmrss_kb,storage_bytes,nnz" << endl;
  vector<string> csvRows;
  csvRows.push_back("matrix,format,mean_ms,stdev_ms,median_ms,samples,vmrss_kb,storage_bytes,nnz");

  for (auto &file : files) {
    vector<int> dims; vector<vector<int>> coords; vector<double> values;
    if (!parseMTX(file, dims, coords, values)) { cerr<<"Failed parse "<<file<<"\n"; continue; }
    int order = (int)dims.size(); size_t nnz = values.size();

    // Build TypedIndexVector for coordinates
    vector<TypedIndexVector> typedCoords; typedCoords.reserve(order);
    for (int m=0;m<order;m++) {
      typedCoords.emplace_back(type<int>(), nnz);
      for (size_t k=0;k<nnz;k++) typedCoords[m].set(k, coords[m][k]);
    }

    // copy values into contiguous buffer
    double *valbuf = (double*)malloc(nnz * sizeof(double));
    for (size_t k=0;k<nnz;k++) valbuf[k]=values[k];

    for (auto &fmtName : formats) {
      // Special-case ELL: do manual COO->ELL conversion (not supported by pack())
      if (fmtName == "ell") {
        if (order != 2) {
          cerr << "ELL only supported for matrices; skipping " << file << "\n";
          continue;
        }
        int nrows = dims[0];
        // Build per-row lists from coords
        vector<vector<pair<int,double>>> rows(nrows);
        for (size_t k=0;k<nnz;k++){
          int r = coords[0][k];
          int c = coords[1][k];
          rows[r].emplace_back(c, values[k]);
        }

        // compute max entries per row
        size_t maxPerRow = 0;
        for (int i=0;i<nrows;i++) maxPerRow = max(maxPerRow, rows[i].size());

        taco::util::Timer timer;
        for (int it=0; it<repeat; ++it) {
          if (cold) timer.clear_cache();
          timer.start();
          // allocate ELL arrays (indices as int32, values as double)
          size_t M = maxPerRow * (size_t)nrows;
          vector<int32_t> idx(M, -1);
          vector<double> valsEll(M, 0.0);
          for (int i=0;i<nrows;i++){
            auto &rvec = rows[i];
            for (size_t t=0; t<rvec.size(); t++){
              size_t pos = t * (size_t)nrows + i; // ELL layout: pos = t*nrows + row
              idx[pos] = rvec[t].first;
              valsEll[pos] = rvec[t].second;
            }
          }
          timer.stop();
        }

        auto res = timer.getResult();
        size_t afterRSS = getVmRSSKB();
        size_t storageBytes = maxPerRow * (size_t)nrows * (sizeof(int32_t) + sizeof(double));

        cout << "----------------------------------------" << endl;
        cout << "Matrix:         " << file << endl;
        cout << "Format:         " << fmtName << " (ELL, row-major per-row padding)" << endl;
        cout << "  nnz:          " << nnz << endl;
        cout << "  maxPerRow:    " << maxPerRow << endl;
        cout << "  mean (ms):    " << res.mean << endl;
        cout << "  stdev (ms):   " << res.stdev << endl;
        cout << "  median (ms):  " << res.median << endl;
        cout << "  samples:      " << res.size << endl;
        cout << "  VmRSS (KB):   " << afterRSS << endl;
        cout << "  storage (B):  " << storageBytes << endl;
        cout << "----------------------------------------" << endl;

        {
          std::ostringstream oss;
          oss << file << "," << fmtName << "," << res.mean << "," << res.stdev << "," << res.median << "," << res.size << "," << afterRSS << "," << storageBytes << "," << nnz;
          string csvLine = oss.str();
          cout << csvLine << endl;
          csvRows.push_back(csvLine);
        }
        continue;
      }

      Format tgt = makeFormat(fmtName, order);

      // Skip formats that include Singleton (COO-like) since pack interpreter
      // does not support packing Singleton levels yet.
      bool hasSingleton = false;
      for (auto &mf : tgt.getModeFormats()) {
        if (mf.getName() == Singleton.getName()) { hasSingleton = true; break; }
      }
      if (hasSingleton) {
        cerr << "Skipping unsupported format (Singleton/COO): " << fmtName << "\n";
        continue;
      }

      taco::util::Timer timer;
      bool failed = false;
      for (int r=0;r<repeat;r++){
        if (cold) timer.clear_cache();
        try {
          timer.start();
          TensorStorage storage = pack(type<double>(), dims, tgt, typedCoords, (const void*)valbuf, Literal::zero(type<double>()));
          timer.stop();
          (void)storage;
        }
        catch (const taco::TacoException &e) {
          cerr << "Pack failed for format " << fmtName << ": " << e.what() << "\n";
          failed = true; break;
        }
        catch (const std::exception &e) {
          cerr << "Pack failed for format " << fmtName << ": " << e.what() << "\n";
          failed = true; break;
        }
      }
      if (failed) continue;

      auto res = timer.getResult();
      size_t afterRSS = getVmRSSKB();
      // Get storage size from one final pack
      size_t storageBytes = 0;
      try {
        TensorStorage last = pack(type<double>(), dims, tgt, typedCoords, (const void*)valbuf, Literal::zero(type<double>()));
        storageBytes = last.getSizeInBytes();
      } catch (...) {
        cerr << "Warning: could not compute storage size for format " << fmtName << "\n";
      }

      // Human-friendly printout
      cout << "----------------------------------------" << endl;
      cout << "Matrix:         " << file << endl;
      cout << "Format:         " << fmtName << endl;
      cout << "  nnz:          " << nnz << endl;
      cout << "  mean (ms):    " << res.mean << endl;
      cout << "  stdev (ms):   " << res.stdev << endl;
      cout << "  median (ms):  " << res.median << endl;
      cout << "  samples:      " << res.size << endl;
      cout << "  VmRSS (KB):   " << afterRSS << endl;
      cout << "  storage (B):  " << storageBytes << endl;
      cout << "----------------------------------------" << endl;

      // Also print CSV line for automated parsing
      {
        std::ostringstream oss;
        oss << file << "," << fmtName << "," << res.mean << "," << res.stdev << "," << res.median << "," << res.size << "," << afterRSS << "," << storageBytes << "," << nnz;
        string csvLine = oss.str();
        cout << csvLine << endl;
        csvRows.push_back(csvLine);
      }
    }

    free(valbuf);
  }

    // Write CSV file into the repository (outDir)
    if (!csvRows.empty()) {
      // create directory (mkdir -p behavior to create parents relative to cwd)
      {
        string mkcmd = string("mkdir -p ") + outDir;
        int mrc = system(mkcmd.c_str());
        if (mrc != 0) {
          cerr << "Warning: could not create out-dir " << outDir << " (mkdir returned " << mrc << ")\n";
        }
      }
      string csvPath = outDir + string("/bench_results.csv");
      ofstream ofs(csvPath);
      if (ofs.is_open()) {
        for (auto &r : csvRows) ofs << r << "\n";
        ofs.close();
        cerr << "Wrote CSV results to " << csvPath << "\n";

        // Try to generate a PNG using the repo's plotting script
        string pngPath = outDir + string("/bench_results.png");
        string cmd = string("python3 ../tools/plot_benchmark.py ") + csvPath + string(" -o ") + pngPath;
        int rc = system(cmd.c_str());
        if (rc == 0) cerr << "Wrote plot to " << pngPath << "\n";
        else cerr << "Plotting script returned " << rc << "; skipping plot generation.\n";
      }
      else {
        cerr << "Warning: could not open " << csvPath << " for writing\n";
      }
    }

  return 0;
}
