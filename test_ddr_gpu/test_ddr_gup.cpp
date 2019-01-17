#include <cl_test.h>

int main()
{
        void *test_buf_f[4];
        void *test_buf_r[4];
        int a =gpu7k_get_viraddr(test_buf_f, test_buf_r, 0, 0);
	cout << "======= SUCCESSFUL FOR TEST =======" << endl;
        return 0;
}

