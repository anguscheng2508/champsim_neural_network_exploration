//
//  Created by Angus Cheng on 08/02/2024.
//

#include <stdio.h>
#include <torch/torch.h>
#include <iostream>
#include "cache.h"
#include <algorithm>

using namespace std;

void CACHE::prefetcher_initialize() {}

vector<int> prev_addrs(20, 0);
vector<int> delta_arr(20, 0);
vector<int> counts(127, 0); //range[-63, 63]
int delta_min = 64;
int delta;
int delta_same_page;
int max_freq = 0;
int max_freq_delta = 0;
int prevVector_reset_count = 0;
int first_iter = 0;
int delta_not_found = 0;
int prev_addr_immediate;


uint32_t CACHE::prefetcher_cache_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, bool useful_prefetch, uint8_t type, uint32_t metadata_in)
{
    printf("\naddress: %d\n",addr);
    // Declare delta_in_range here to reset it for each iteration
    int delta_in_range;
    int page = (addr >> 6) >> 6;
    
    //Starting at the end of the vector, then decrement - proper methodolgy. Allows for the edge case i=0 to be dealt with
    //This for loop is responsible for recording the previous 10 addresses and storing into an array
    for (int i = prev_addrs.size()-1; i >= 0; i--) {
        if (first_iter == 0) {
            break;
        }
        if (i == 0) {
            prev_addrs[i] = prev_addr_immediate;
        }
        else {
            prev_addrs[i] = prev_addrs[i-1];
        }
        
        delta = (addr - prev_addrs[i]) >> 6;    //Calc delta
        delta_arr[i] = delta;
        printf("Delta: %d\n", delta);
        int prev_page = (prev_addrs[i] >> 6) >> 6;
        
        if (delta_not_found == 0) {
            if (page == prev_page) {
                delta_not_found = 1;
                delta_in_range = delta;
                printf("Delta_in_range: %d\n", delta_in_range);
            }
        }

        //We take absolute value here to get the delta closest to 0 i.e. closest to current address
        if (abs(delta_in_range) < abs(delta_min)) {    //If calculated abs(delta) is < than abs(minimum delta)
            delta_min = delta_in_range; //then assign delta_min = delta_in_range
            printf("Delta_min: %d\n", delta_min);
        }
    }
    
    delta_not_found = 0;
    
    
    //First instruction will have no need to store previous addresses - because theres none
    first_iter = 1;
    prev_addr_immediate = addr;
    
    //------------This section has no relation to the vector and deltas of interest------------
    if (delta_min > -64 && delta_min < 64) {
        counts[delta_min+63]++;
    }
    
    if (counts[delta_min+63]>max_freq) {
        max_freq = counts[delta_min+63];
        max_freq_delta = delta_min;
    }
    delta_min = 64;
    //-----------------------------------------------------------------------------------------
    

    printf("Prev vector: \n");
    copy(prev_addrs.begin(), prev_addrs.end(), ostream_iterator<int>(cout, "\n"));
    printf("\n");
    
    // This prefetch is not 'used'. Only interested in the deltas printed (in the above sections).
    uint64_t pf_addr = addr + (max_freq_delta << LOG2_BLOCK_SIZE);
    prefetch_line(pf_addr, true, metadata_in);
    return metadata_in;
}

uint32_t CACHE::prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in)
{
  return metadata_in;
}

void CACHE::prefetcher_cycle_operate() {}

void CACHE::prefetcher_final_stats() {}


