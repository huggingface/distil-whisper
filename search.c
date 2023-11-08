#include <ftw.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *search_file;
int search_folder = 0;


int display_info(const char *fpath, const struct stat *sb, int tflag, struct FTW *ftwbuf) {
    char *dot = NULL;
    if ((tflag == FTW_F && !search_folder) || (tflag == FTW_D && search_folder)) {
        char *found_file = strrchr(fpath, '/') + 1;
        if (!search_folder) {
            dot = strrchr(found_file, '.');
            if (dot) *dot = '\0';  // Null-terminate the name at the dot, effectively splitting name and extension
        }
        if (strcmp(found_file, search_file) == 0) {
            if (dot) *dot = '.';  // Restore the dot before printing
            printf("%s\n", fpath);
        }
        if (!search_folder) {
            if (dot) *dot = '.';  // Restore the dot in case the path is used elsewhere
        }
    }
    return 0;
}



int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <file_name> [--dir <dir_path>] [--fol]\n", argv[0]);
        return 1;
    }

    char *dir_path = ".";
    search_file = argv[1];

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--dir") == 0 && argv[i+1] != NULL) {
            dir_path = argv[i+1];
            i++;  // Skip the next argument
        } else if (strcmp(argv[i], "--fol") == 0) {
            search_folder = 1;
        }
    }

    if (nftw(dir_path, display_info, 20, 0) == -1) {
        perror("nftw");
        exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);
}