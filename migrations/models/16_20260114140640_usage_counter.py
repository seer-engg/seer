from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "llm_usage_records" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "workflow_run_id" VARCHAR(255),
    "provider" VARCHAR(50) NOT NULL,
    "model" VARCHAR(100) NOT NULL,
    "input_tokens" INT NOT NULL DEFAULT 0,
    "output_tokens" INT NOT NULL DEFAULT 0,
    "total_tokens" INT NOT NULL DEFAULT 0,
    "cost" DECIMAL(10,6) NOT NULL,
    "operation" VARCHAR(100),
    "metadata" JSONB,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_llm_usage_r_workflo_569a86" ON "llm_usage_records" ("workflow_run_id");
CREATE INDEX IF NOT EXISTS "idx_llm_usage_r_model_6df62f" ON "llm_usage_records" ("model");
CREATE INDEX IF NOT EXISTS "idx_llm_usage_r_created_ea395e" ON "llm_usage_records" ("created_at");
CREATE INDEX IF NOT EXISTS "idx_llm_usage_r_user_id_7b16bb" ON "llm_usage_records" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_llm_usage_r_user_id_f9c98c" ON "llm_usage_records" ("user_id", "created_at");
CREATE INDEX IF NOT EXISTS "idx_llm_usage_r_workflo_eff0ba" ON "llm_usage_records" ("workflow_run_id", "created_at");
CREATE INDEX IF NOT EXISTS "idx_llm_usage_r_model_ddd30b" ON "llm_usage_records" ("model", "created_at");
COMMENT ON TABLE "llm_usage_records" IS 'Detailed log of individual LLM API calls for cost tracking.';
        CREATE TABLE IF NOT EXISTS "usage_counters" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "resource_type" VARCHAR(13) NOT NULL,
    "period_start" TIMESTAMPTZ,
    "period_end" TIMESTAMPTZ,
    "count" BIGINT NOT NULL DEFAULT 0,
    "value" DECIMAL(10,2) NOT NULL DEFAULT 0,
    "reference_id" VARCHAR(255),
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_usage_count_resourc_0fc711" ON "usage_counters" ("resource_type");
CREATE INDEX IF NOT EXISTS "idx_usage_count_period__17e3be" ON "usage_counters" ("period_start");
CREATE INDEX IF NOT EXISTS "idx_usage_count_referen_dbccbb" ON "usage_counters" ("reference_id");
CREATE INDEX IF NOT EXISTS "idx_usage_count_user_id_a31482" ON "usage_counters" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_usage_count_user_id_25ba06" ON "usage_counters" ("user_id", "resource_type", "period_start");
CREATE INDEX IF NOT EXISTS "idx_usage_count_user_id_6b433e" ON "usage_counters" ("user_id", "resource_type", "reference_id");
COMMENT ON COLUMN "usage_counters"."resource_type" IS 'WORKFLOWS: workflows\nRUNS: runs\nCHAT_MESSAGES: chat_messages\nLLM_CREDITS: llm_credits';
COMMENT ON TABLE "usage_counters" IS 'Tracks usage counts for various resources per user and time period.';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "llm_usage_records";
        DROP TABLE IF EXISTS "usage_counters";"""


MODELS_STATE = (
    "eJztXWlz4jq6/iuqfJl0Fd2TvXtS99wqAnQOc8hygXRPTaeLErYAT4zNeEmamTr//Urygh"
    "fZkYwBGzQ11YfYemX70fo+76L/Hs1NFen2pycbWUfX4L9HBpwj/CN2vQGO4GKxukouOHCs"
    "04IuLkGvwLHtWFBx8MUJ1G2EL6nIVixt4WimQYq2IZaCNgK0GjAxLQBdZ4YMR1Ogg1RA6/"
    "pEKlNNBdemGVNBOdfQ/u2ikWNOES5APunHT3xZM1T0C9nBn4uX0URDuhr7Yk0lFdDrI2e5"
    "oNe6hvOVFiSvNB4ppu7OjVXhxdKZmUZYWjMccnWKDGSR98LXHMslQBiurvuABdh4b7oq4r"
    "1iREZFE+jqBE4inUIzuBgByr+kmAZpCfw2Nv3AKXnKx7PTi88XX86vLr7gIvRNwiuf//Q+"
    "b/XtniBF4H549Ce9jxvBK0FhXOFGwB+xwGvNoMVGLyKSgBC/eBLCALCdYjiHv0Y6MqbOjA"
    "B3eZmD2Ldmv/V7s3+MS30g32LiQeENlXv/1pl3j8C6ghHNoaaLgBgKFILQByhEMCiygnA1"
    "iDeB4fnZCQeGuFQmhvReHMOJZtnOiP4lAGRcqpZobqRH6rAAmDEhiWWApaJDbW6ngfz74O"
    "GeDeRKIoHik4E/74eqKU4D6Jrt/KwkpjkQkm8m7zy37X/rUeSO75r/SILa6j3cUBBM25la"
    "tBZawU0CYFubGu5iZJuupQh12JRgLTvtJc9sepk9mV6m5lLFQuRzR9BJo4k3ZMjR5iij58"
    "YkE3Cqvuin4EeR5X4L6OJvUB8Mfem3bQ66w+5dZzBs3j3G+nS7OeyQO2f06jJx9fgq0RJh"
    "JeB7d/g7IH+Cfz7cd5JdPyw3/OcReSe8DzZHhvk2gmpk2xNcDYCJNay7UAs2bFxSNuxOG5"
    "a+PFFmJi+RbTm5MIbKyxu01FHszqoDmER5ws1qGEgh8DGWpRu/iq9/9JEOKcbpFvdVxIcm"
    "rq4V1lbNZv8z6MvB1VXzRzRAw0FTi37uCDcLXRHWBKe7qrKPVmvMHgBkIzzPO+XBM6D11R"
    "gcXMV0ivVa2x2HD1kTnqFX5SBSY40BejOtl4luvq0Jyne/mhojocyggweQba/fRQI08AbX"
    "GXg11hiYYIcRdJWRasHJupNMAFGb1FXJvbzQ8BktLBPvHKBeEiyPfnU17jYhNBZSTEstCZ"
    "g+rWwvYHHLmmb6bp2nl0AzDYF5RVaJc/A3r7b6TjG6Ph+5NpyicgZSr3f3RGqr/UDyMFFM"
    "F29XrTVBoYi0vKpqBglRLs0zM0vdjN9aoTfWdB0/j6xbE03P06geDDQ08T/vo3jj1fnoVV"
    "mvEUdAm5/NEzDOoYE7hupXR4Qz9GuGlZahgmcbbJnqfyHjre2YBAFAHw8iNf7VMV+QwWfJ"
    "5aqEYdb9Qa2ItIdY5qumxn+PoEKHK7Ey/lzHAnyjTffICPy3s7Pz889nJ+dXXy4vPn++/H"
    "ISWoPTt/LMwjfdW2IZbkSprvdNxdGW4uXnY61birG4/tw8q5cXgDQhXk90N2Kuw8BgXXpE"
    "Z6ARMpQ0ukP0K2NWYMnWBdo8Nrzzj2G+1S4kw3sP97dB8aQpL46zhSYWsmfFgGYK18R6t2"
    "2g0a+FhsEqYPSJS5Zg9KkW2BWy8QSfnWu9sxVzwdpCZw+TlYQcG8yx4c0fFBWBRTQutb0J"
    "/ugGQcvbDlV9dzJHDiR7QRG3F6aw9IBJzCRMDxgHOi5jZshxfQkltth9sZKpvaLSui+XG2"
    "GOF2HaiVA6vuyFf4R0fNnThmWwsxkO6JnETbb/OYO9qUjzleDFn/IWimOYBvCraSFtavyB"
    "lhTHLn4jaDC9JBPhItXDL4uNxZct+BaSgdGugT8PfxRyvFW0OWg1252jP3k8rHbqPFQhpj"
    "u2PdmVv1CF8BAyoAiaCVidhWEqyOhT2eaCTKe4900Gj8QeapMArUAQjDHqhO13ZtABM2hM"
    "bWBOJgAaKf4/bTxYr7pn49no/ILzhY7s62cD4P99BAN34dkjsNrxL1wQ2HjAK+QRcOIgy6"
    "slLKzjhgLEdmwvIH74X4GCH2kgPXiNoOCt5vzujvFb4nVOc0xribUU4wVX6pjk1TQyjeqr"
    "rs40biStNKRg8Nmh1hleYFg52PaReBU/pSlEmkLqbQpJjQleMFOCEtEEomJGpYRYPdHciD"
    "EpROYFLQsh6svVhL7dAqKiQYkyHjGzTxbhaJnCkqNtSI5WcrSSypMcrWzYFPuTVGeZm8s8"
    "BTOjgkL07Q5W9C3qnJIYl8T4LojxvOFeAoLiYcbVIX6TWGZMZsUNDpJc3xq57kOTT62v8O"
    "Mj1iPt9z6tfkuaRFPAK8GT+tAbpvHRJ7wtpJK8aFgSOJpPOUdocIBLB+pbmmEvrWZCtg8c"
    "PMRtgAzFWi4Ipf4KdRdfOKb/Jf6aH8Cze3Zyeh6UIbXocIksQuWrOhYZL4ECdT0nuxubKa"
    "ctg1v0R6irrq5y8eReg0iWXLLk29nEbJjTjXZnASgTYhLNHdCPO8fv9IQHQFwqE0F6Lw5h"
    "uAakccz2ZI4J1QXMbXszeyBN8Csha2H5iwtvT2UK15I1v7rg6LRXF5l9ltyKA1uEKZcEuS"
    "BBLuNUqk/38cWpSEuHtHSsP6fvCyEuLR172rDS0lEFS0euz1Ie2PleSxJkaU5qSHNSgT2s"
    "NCdV1ZzEmjZLwLPusSpJTBMLg6hpbpPGKD8rbOcVGUw7VOx+I88EFaSsRaQop/Xp3rTmUN"
    "f+g1SgGYo5p+EXXj2A1pM2K/GJMI07wRsGLrBBdHh6exTcoRXSgAhi/uGW98Rm0J4xLEQr"
    "1dRCCsIKI918CzxAzIS0R/ajjZ06lMCdl2RINldNSOQteARnDw3OzpkzNmu1q16zyzIQDa"
    "ckgY7KFK4lE7+Z07JW07UAqHGp3aJ51Ma7CQuvhZrtaAogL0XOsVPB2wwZINX8QLOBa8BX"
    "qNE3SS2yO2sJU1FcyypEZCVEJWe/Y84+urkRbMqEqCQlK8Y2o0AZ4LVchgIlmC2rZXXfiN"
    "2S6I4LuNRNyFjlcyKp4mLSRMwBdZ5lsWO48xRhsXsrYzA7pu2MR/1Oq9P91mlfg6DQs9F/"
    "eBrSK6brkL8f+w+tzmBALuF9AckuSq5+bXZ75NIEbwn8qndurUSWZTL4zZxpJhCQXT+j66"
    "fIZK4ENFs+8KE6PN5G3cBZ5zFlE3DJY5ve5+FSR0e9T8f5zyKO2BNt6noMLICOA5VZ4KQd"
    "HHeRZuaEpd8/X/tHlNoPD9oQ5+MMAo5kziRzVnkuQjJnZTNnweBPr2CmqSNoZOwlVlIJ3M"
    "ZYbFPdMxz9Ze8gbh4eerEdxE036f77dHfT6R+f0s6KC2meeSqN50TT2QfXZG/MIiJya8ah"
    "lfgpwIQwjspIkDlAjk6UeL8igjVDVELOAXkQAkSynYtsF5Jy0nSxOkR34TqkG3o7e4E+nJ"
    "asUhc+auPn/sBgNkCXvGgbTX4CLKMZxN4faAKAfoTNR1dsu7NPTGuOlbDJRPsl0tcTYju2"
    "K7Vc2zHn4KnfA7buTmkYL3nDwOXCBsfo0/RTA/yF9CTckT6Su3/5UIRB2khsGIXT11xEdi"
    "xxsUoNjaZlwSUwJ97IoO/v69x2un0qPDbE192EWKVa5StB/ambYD+OHc3RsXikcAPYzpKc"
    "N8g5SLa+LzJ1fUTyCVivUCfJBEyDNXayFdAs+e157V6drLFol6x+GuiXM6KYiBtCk7K7so"
    "Qe/c/ENbzEDDZeuJBKnkhzHX8KSTmfrqSmRNdR/pdhnrjHnwNsQga6JB8D+TBAXhscPw1b"
    "Hzi9EGpiR+WykdOmVVzLNq3Rv2yWK3KODsKQrdKEWFklhAInHlGYENuiwc98SY+lgjrI+R"
    "nHDuv8LHODRW4x0KSGt2I9OC4qOzBvByY2KHMyKbo+M8S3tzxXaHWmYOim8jIy3wzB1DBp"
    "UclLpHEtHoSfVYd07dvxtkWGke+pY58MI9+LhpWHmpWxNYi6m/ADl5A6JPBkpHLJkcpBXy"
    "oBue+RquqLXmJwrXGmnnRp3IRLY9jLGH6M0R6Y7bwYtDBv/DB6WxkDSUJZZ+kd34ZVMhuo"
    "Fpw4dgO8krPeMBANAA0VLNyxrtnEI5GwOYzEtWVV+r57o3RHLLRGZ7sjHlYyzY1o7tE3Sy"
    "GZnU4zIVYTJiRvi7+JfJoOFHPqCspLXrTxPi9K8mOKgBuUl+BygCs5n72gBiTns6cNm+J8"
    "wh3pyN+pCoYRZIgfaBCBZNAkCbQbEig1EEtkg76taqzcCOYFM2uiKk4OZQZQFmeLMoI469"
    "OF43vBGXRGNrLt9YEJeiJW0Z2BV2ONgVlYJl7EoV4SKI9+dTVG5ICJ1tg5Gj5ZWA4U/NN2"
    "lbqFEPEc4aMI35qN24OBhib+hx+9dlBhfbpSQRqeTKp3eFKFdF+XychHizV4yPkRXQDmng"
    "gnU981VO1VU12oA18QaAY5840Q6/5akqbiuaUk1751rt0ydSGuPShfU669/MM4TMNh5tHK"
    "JtkjInVBcesM+0wzXsgLCcAalZGGCyastov1F5uwfUjVWGfDZtPsDFHJuDf4zBnyiDBp1p"
    "DsdxGzhr85FONs40KHRNuyWAzhvENRqVLsBHXYIucQ3vaK0CqJsK0tTdZIsLbxkbb+gRjp"
    "3ptGPaAKBDEXYOG23X+5OfL42OSAuxQ3wGhvfYd/iHRqXv4hSkC/zz+0IowBzZIQevhBfI"
    "30BcZZI7xCkn3YfuLBGdm7CJ6DEBMqR4MuiONRDxrTWwsuZsB7KdBtJ7vfzlwAac4KIVwD"
    "gR1nq3mgvyDJKoBfiI5YPB5BZB2uBL5S6dhTpUP6Uu1Fw8r4ORk/Vy1NUkaCFYkESw7hEt"
    "A7AOczHhepqO23HOefiAG6PthWxPmnYp4Km44u9Bw4cgiF0MODg0rwwvn4OIQ7l8p7IYBe"
    "oB/VMfIOReCUkQzC1hkEe4EUITumX16eYcVjWLPQq8YmvzP7ZlRke/u+0133UKk+7rv6+A"
    "p1TaXLvpfuTsh9giksbfscU1AEObz5MEQPVMgQl9BzQB/MR+OlIG2SlDvQcLMKcSd12Obl"
    "BZ2FPWp72n919LFGUvlPjq8yLe/ZJFVBy3sVu2upBFX5unHIGuSox1FmgUNDjvEa7yvJfb"
    "yLR29U510l0VE1BwT1pHVkPhGpIm9fRXbnc2gJnewXEamLi3rc4nt5ypMjG5fKtPjSewn3"
    "ack17OgQ5wx8d5HHfYHoCW1HZfXU8kNSFhadiUdT4pgi0l9TglJL4ui3ePHVNbwbE4Y7JS"
    "jh5oBbBlRsAWQVKZpaiMOMS8ok7zLJuySjpZOabFheK0Mw7EQ515TcIflcVSFkTNLVB9ft"
    "pKvfBl39dh12V136nyfqjr2kbM94UuGemFop17ee5FgAUt6W6TYongRJzOWyOj26aCqkPl"
    "JMK24UYJdocNlELFqY95AC05pDXfsPUlPHCpCmx5fHSxC8xmCBFEAUa8a5BMXrkbYTeRTB"
    "Rq0m8iiCTXJHG8noI01Sm/Q9y8rVmzl1RiQO0vlVHo0hj8aoJ7g6tB2M3Xyh6WjkHdad2K"
    "abpo6gwYaaIZ1AfYzFNzUBsLesZUB98/DQi0F9002uSk93N53+8SnFHRfSPD0qPTVIU8Re"
    "MNbSFLGnDSvj5eVpGZUMWN5GNCrJS59HK7m8Sa2CVPnvE0qPRFUgGVVXPBAWBoGvCjg2TO"
    "BgnPAFa/khTSQVkJcEkoxP3atNu2YsXLE8xisJqRVxAIwfP9EYabizAV5JSIA5ALZN11Iy"
    "KOaO4c5TG4T4bBFKb9EDGi+mrhd3ksys0Lx/avaugVfg2Rj2u7e3nT5pRnpwUnIJ2417dJ"
    "6fOQfiu/A5x0VdpDIQ/7+nzlOnfQ28As9G/+n+vnt/e012AiTW9dkYPLVanU6bFLJdRUFI"
    "JeW+Nrs9cmkCNZ383Wretzo9ekkhn63r/uN23lqm6+AJW2QCWknICYhjAqLh6CKGmVBAmm"
    "Qyzq7YSU7THfifyRybkloqyBniZdQq1rBxSRlgsOMAg4lmeOd2irdkQlQ25Y6bkhBFmiKk"
    "TUdE5GaLY7MVPYxW0EM9LXmgburBwb7oFRmOGIgs0QNFURp3ahUhUf8OF6IQP+BbFMO49A"
    "FhKY2MJRgZWR2yBOQEQnSq44SfBE84Qic5JksEkv9s7BrgGZ+yOCKfIlu9EjAdepueQaLW"
    "2uLK2Am/j2ls51ceqJ2gutqiydoSV9FBI5gRcpw0IpMGh6OGPyp5Dz6fz/0U4MSyEk9y5t"
    "fEOPecU4jhlvEjOR0Hk4jhzse4yE/pt1HG9jTHb6N+VsJ2v/l1yDAS0uvXgP7n2eh3ep3m"
    "gFj6gl/PBiE2ut/IteBXNYx/0nlmo74dvolkYplz7wSFUYF87+/UckAamjRcHYDhCm9UtA"
    "myhTwSojKSJueYmcg8PppBm5FOLycbZFSonvG3VxccK+jVReYKSm4xIxiDbSP/tJ4W3B7p"
    "e1K9eXw36Zjqvw7KnELl0bwyp9C6jOUuMuFUl/pZOxFOJOWwO9Y9X46gTUo6P69WAMf6Wh"
    "AItD4GfvBRfWDYJB94o+k6fqlHy5xo9LtTdGCiRCOPDRx7ZcnpCKQwJxnoPwH4Uv5JgAu4"
    "JBf9XD7HGHftVVNdqANilgLkFHME54xArvWrk3FdW+cH6denkONjBwPZLXKDq+7DIAi79+"
    "3ut26bxm6sCj4bw07z7pp2syJ04CkPHXiaTQeeMiI38EujkeLajjnPcF/JOy2AJV2Kr/XG"
    "O7P0tJaElczOcMgNm9pfmm8Y65G4G19K7oAPmBMxabOQT8MueEZaUZerqpyPlupMwiek8b"
    "q9iCdJ9bfVtfV7EU2SyvrebPUoCcv7OlK0fTgVpehDgJdtFTgawv+YWMMZx/WetGIkLi4V"
    "oR04StBt9buRDe9uzN+PcDjYzTnu9Iy1hlPZ1Jjmm00qmxMLIYaa+bXf6eDy+Oaz8dh/uC"
    "Yjl/4aPfaeBvTP0UJ37WfjqTfsN68Brs6CRVTPLxxN8CWzAb7sQcYAPC1rr6xGaLaG3W+4"
    "GbwCQeT/KvCfZAN4bA6Go/YTLrUgaQ1VF9FcDs0eTS2AGwLqNLdA9771cPfY6ww7hDcg2Q"
    "/pyl+gwcrnChTXsohH4wJZmqmOaNSosGabUYeMVNz1qVbxhkEGY70Ralq/Btmwu25YOgeN"
    "YH7T5mZizapCpmOVhN8e8kKS8NvThk0RfglbpRjrxxaW1J8g9ZeAcW3+L22rrhr4vEwgu4"
    "MJ04GCnNeAUgff0Xhmmi9eMBaD8mKUauQxXj4h8eYJeKFRnIxX27VonI/t4Cli6tnyvccD"
    "vzrgVUc4LOTFBGkqmi9MB18lKrBCTvcypmk2rNyqJVO2daYsO2FFNjeWk6miIH2wD3RYpu"
    "8FG8Tt+1tUHsIFXOomZHTF7CiGiIiMr0pu75hRDPWjDS2kIO2VmWq032l1vBjBoBDlblud"
    "wYDSgqv1JbxOCvuX00lHq0ETQsfBKyQrhXTmihYVOciYCHrWjXDa0LhUTVJc5mmPm8gdKk"
    "mZvdDdJSmzpw0b+mUU5hLK0317vbsncvJv9qG4iRKNPJ1X1+cjlxQWOxW3jRy6mgPdnAJz"
    "EnEfBvjpoPnYBQrUdZtqqwqGHpAKX9gKbvG6no1noznFzTol/R5LYhWYfnrLdA0HWVRE1+"
    "aaA5QZojINMHaJVoxsZNHNDCmiBq+AQdeXjqbY+JeKL4/d6ZRfdf4RzUQVmZlxsR+xo2Iy"
    "itD2Sd6Q+nihfUu2Ps5oCF6NkiG6oR1N/bRzvOPHozbLYSVDt4zI1FNLv+RRTS6zVZPLlG"
    "oSTgK8IIYCm0Jwo33x9IRPucvT7lIY0tN+8Ez9gljBeNmTZELsINU87xwNcfBScgeJnmM6"
    "UBcHLyl2kNiRvR1DTUKKNod6hvprMtOrqJ7MJ1+2mgtJDjbtTqt71+zhqa1xlfDoCGbEi/"
    "SROQvygUwv+uzFIyZUE3ZmCwtIcJqiCFEelZHpfhr8icjqwXqVvQvaF26EQXpVMcd8BZXU"
    "nJwvlUvtvSP3kooeHxzlmI4YDFzsfiOPf/O4N8Urykm+DQn9ZQMqCqiox429QkszXRvgkU"
    "zPh7QBXty9tBGEzSJTIfCcY9MkXAl1EjJu4C4WpoVlx6YzA3RXG9R2DHX9Iyn/gUq+4b5l"
    "viE1vD03DWemLz/getrIxt0c3yNvgCYTTdGIKwvumxZNjkHk+0jVbKBAZRYQgZ1fkEQi0G"
    "Iasq+fjY9gSN8gTE5DKyQvfx1+EB0yv31/6P/xtffwfdAAUa//3+5NA5F6+q5hA2eGn0hf"
    "Mynef7pPSp6dnF18PDnF/w9vIEP1L5/hy6RavC9zwBzZBHXv5YJXTT4Br7XDEZ6ZB83bDn"
    "6UhSbIQnjs48HxWyIfkiBPGXsOnUqiYQ+Ul8wrHX0TSVaWTlam8C7iU5Bu4p0yRUfhaLte"
    "DU16iOmAnmCKf8e6+zVQ8DgZBePk2ej17katfqfdHeJ7xIKB94Oq5pnmhdWHcx7t4TxbeT"
    "hPEaFrhB7tJORoM2t71XeyXIEpxSONZIRRpRqS7jLSbXijTbNzeQYidSLj/nZ2dn7++ezk"
    "/OrL5cXnz5dfTsIVKX0rb2m66d6S1SnWYGnK7hXqLmNZyuXsQpndkHYnn9ZAmY+zO+Pm7G"
    "K7J+bqzsYwKSetkLXklKTDjfSkOqyGTYW3SbJQkoX7TRY2sSagzI4YNKF/p5FHEMJVmfeI"
    "wWxAZdTX1ombzMMRszd1EZF6OkNtZEdHhoYAiH7xegK4EVs2fqLDPP0w25QdEZExX1mm7J"
    "36g//5/7yVCzI="
)
