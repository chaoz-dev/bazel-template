#include <gtest/gtest.h>
#include <glog/logging.h>

TEST(GoogleLog, SanityCheck)
{
    LOG(INFO) << "Hello World!";
    EXPECT_NE(0, 1);
}
