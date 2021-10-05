#dicttionaries for the PHANGS muse sample
# v19.0
# Added last 1385 P04 pointing and 2 P03
# + updated 1566 removing the old P03
# Added 4321 P08 in Feb 2021
#this version also includes a dicttionary for the specific filter to use for alignment puproses 
#this version includes the updated exposures to combine for DR2 (excluding the ones we decided to discard)
#exposures to discard have been added by Eric E. wrt version12 of the dicttionaries
#two missing exposures (not spotted during DR1) have been added to NGC1365 (exp4 of P09 and P11) 
# These two extra are actually BAD so we remove them
#updated on 12/05/2020
phangs_muse_sample = ["IC5332", "NGC0628", "NGC1087", "NGC1300",
                      "NGC1365", "NGC1385", "NGC1433", "NGC1512",
                      "NGC1566", "NGC1672", "NGC2835", "NGC3351",
                      "NGC3627", "NGC4254", "NGC4303", "NGC4321",
                      "NGC4535", "NGC5068", "NGC7496"]

dict_phangs_muse_sample = {
       "NGC0628" : ['P000', {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1}],
       "NGC1087": ['P100', {1:1, 2:1, 3:1, 4:1, 5:1, 6:1}],
       "NGC1365": ['P100', {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 30:1}],
       "NGC1512": ['P100', {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 30:1}],
       "NGC1672": ['P100', {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1}],
       "NGC2835": ['P100', {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 30:1}],
       "NGC3627": ['P100', {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1}],
       "IC5332":  ['P101', {1:1, 2:1, 3:1, 4:1, 5:1}],
       "NGC4254": ['P101', {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1}],
       "NGC4535": ['P101', {1:1, 2:1, 3:1, 4:1, 5:1, 6:1}],
       "NGC5068": ['P101', {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1}],
       "NGC1566": ['P102', {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 30:1}],
       "NGC3351": ['P102', {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 30:1}],
       "NGC1433": ['P102', {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1}],
       "NGC1300": ['P102', {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1}],
       "NGC1385": ['P102', {1:1, 2:1, 3:1, 4:1, 5:1}],
       "NGC7496": ['P103', {1:1, 2:1, 3:1}],
       "NGC4303": ['P103', {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1}],
       "NGC4321": ['P103', {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1}]
       }

dict_expo_NGC0628={
 1:  [('2015-09-15T05:00:21', ['1', '2', '3'])],
 2:  [('2017-07-22T07:36:21', ['1', '2', '3'])],
 3:  [('2017-07-25T07:31:28', ['1', '2']),
      ('2017-11-13T03:43:40', ['1', '2', '3'])],
 4:  [('2017-09-16T04:17:06', ['1', '2', '3'])],
 5:  [('2016-12-30T01:01:19', ['1', '2', '3'])],
 6:  [('2016-10-01T04:56:00', ['1']),
      ('2016-10-01T05:21:15', ['1', '2'])],
 7:  [('2016-10-01T06:08:00', ['1', '2', '3'])],
 8:  [('2017-07-21T08:25:39', ['1', '2', '3'])],
 9:  [('2017-11-13T01:22:29', ['1', '2', '3'])],
 10: [('2014-10-31T03:39:46', ['1', '2', '3'])],
 11: [('2014-10-31T04:40:25', ['1', '2', '3'])],
 12: [('2017-11-13T02:32:55', ['1', '2', '3'])]}

dict_expo_NGC1087={
 1: [('2017-11-13T04:56:31', ['1', '2', '3', '4'])],
 2: [('2017-12-21T02:05:40', ['1', '2', '3', '4'])],
 3: [('2017-12-21T03:09:30', ['1', '2', '3']),
     ('2017-12-21T03:56:29', ['1', '2'])],
 4: [('2018-01-12T01:32:38', ['1', '2', '3', '4'])],
 5: [('2018-01-10T01:43:24', ['1', '2', '3', '4'])],
 6: [('2018-01-11T01:02:44', ['1', '2', '3', '4'])]}

dict_expo_NGC1365={                                          #this is the target causing memory problems for the combine 
 1:  [('2018-01-10T02:49:23', ['1', '2', '3', '4'])],
 2:  [('2018-10-17T07:19:24', ['1', '2', '3', '4'])],
 3:  [('2018-01-20T01:16:28', ['1', '2', '3', '4'])],
 4:  [('2018-01-20T02:25:06', ['1', '2', '3', '4'])],        #there is a tmask foor exp 1 TO BE USED
 5:  [('2018-10-16T05:26:10', ['1', '2', '3', '4'])],
 6:  [('2018-11-05T05:41:35', ['1', '2', '3', '4'])],
 7:  [('2018-11-06T05:32:56', ['1', '2', '3', '4'])],
 8:  [('2018-11-07T04:30:38', ['1', '2', '3', '4'])],
 9:  [('2018-12-04T03:58:31', ['1', '2', '3'])], 	# 4 is bad
 10: [('2018-12-04T04:53:10', ['1', '2', '3', '4'])],
 11: [('2018-12-05T04:08:13', ['1', '2', '3'])],        # 4 is bad
 12: [('2018-12-05T05:18:24', ['1', '2', '3', '4'])],
 30: [('2014-10-12T04:31:28', ['1', '2', '3', '4']),
      ('2014-10-12T05:30:02', ['1', '2', '3', '4'])]}

dict_expo_NGC1512={
 1:  [('2018-12-30T01:11:52', ['1', '2', '3', '4'])],
 2:  [('2018-12-30T03:46:33', ['1', '2', '3', '4'])],
 3:  [('2018-02-17T01:02:45', ['1', '2', '3', '4', '5'])],
 4:  [('2018-02-18T01:08:42', ['1', '2', '3', '4'])],
 5:  [('2018-02-19T01:04:07', ['1', '2', '3', '4'])],
 6:  [('2019-01-10T02:41:43', ['1', '2', '3', '4'])],
 7:  [('2019-01-10T03:47:10', ['1', '2', '3', '4'])],
 8:  [('2019-01-10T04:52:57', ['1', '2', '3', '4'])],
 30: [('2017-09-21T06:53:05', ['1']),
      ('2017-09-21T08:30:27', ['1', '2']),
      ('2017-09-22T08:39:40', ['1'])]}

dict_expo_NGC1672={
  1: [('2017-11-12T06:54:01', ['1', '2', '3', '4'])],
  2: [('2017-12-23T04:11:46', ['1', '2', '3', '4'])],
  3: [('2017-11-13T06:07:01', ['1', '2', '3', '4'])],
  4: [('2017-11-25T05:07:09', ['1', '2', '3', '4'])],
  5: [('2017-12-26T05:11:09', ['1', '2', '3', '4'])],       #there is a tmask foor exp 3 TO BE USED
  6: [('2017-12-19T04:31:59', ['1', '2', '3', '4'])],
  7: [('2017-12-19T05:38:10', ['1', '2', '3', '4'])],
  8: [('2018-01-11T02:26:31', ['1', '2', '3', '4'])]}

dict_expo_NGC2835={
 1:  [('2017-12-15T06:22:14', ['1', '2', '3', '4'])],
 2:  [('2018-01-16T07:38:48', ['1', '2', '3', '4'])],
 3:  [('2018-01-18T03:42:20', ['1', '2', '3', '4'])],
 4:  [('2018-01-23T03:26:36', ['1', '2', '3', '4'])],
 5:  [('2018-02-14T02:03:35', ['1', '2', '3', '4'])],
 6:  [('2018-02-20T01:20:57', ['1', '2', '3', '4'])],
 30: [('2017-02-02T02:58:32', ['1', '2', '3', '4'])]}

dict_expo_NGC3627={
 1: [('2018-01-25T07:19:09', ['1', '2', '3', '4'])],
 2: [('2018-05-13T23:25:01', ['1', '2', '3', '4'])],
 3: [('2018-05-08T01:35:58', ['1', '2', '3', '4'])],
 4: [('2018-05-14T00:35:00', ['1', '2', '3', '4'])],
 5: [('2018-05-14T01:41:04', ['1', '2', '3', '4'])],
 6: [('2018-05-14T23:25:02', ['1', '2', '3', '4'])],
 7: [('2018-05-15T00:29:52', ['1', '2', '3', '4'])],
 8: [('2018-05-15T01:34:18', ['1', '2', '3', '4'])]}

dict_expo_IC5332={
 1: [('2018-06-14T08:00:41', ['1', '2', '3', '4'])],
 2: [('2018-07-11T06:07:18', ['1', '2', '3', '4'])],
 3: [('2018-07-11T08:14:50', ['1', '2', '3', '4'])],
 4: [('2018-07-11T09:18:27', ['1', '2', '3', '4'])],
 5: [('2018-07-12T07:19:22', ['1', '2', '3', '4'])]}

dict_expo_NGC4254={                                          #AO
 1:  [('2018-04-16T02:49:03', ['1', '2', '3', '4', '5'])],
 2:  [('2018-05-19T02:22:33', ['1', '2', '3', '4'])],
 3:  [('2018-06-08T00:17:56', ['1', '2', '3', '4'])],
 4:  [('2018-06-08T23:17:55', ['1', '2', '3', '4'])],
 5:  [('2018-06-04T23:35:32', ['1', '2']),
      ('2018-06-05T00:11:22', ['1', '2', '3'])],
 6:  [('2018-06-05T01:06:43', ['1', '3', '4'])],             #removed one bad exposure
 7:  [('2018-06-09T23:26:07', ['1', '2', '3', '4'])],
 8:  [('2018-06-06T23:44:42', ['1', '2', '3', '4'])],
 9:  [('2018-06-13T00:04:09', ['1', '2', '3']),
      ('2018-06-13T00:51:40', ['1'])],
 10: [('2019-03-11T04:59:39', ['1', '2', '3', '4'])],
 11: [('2019-03-02T05:27:50', ['1', '2', '3', '4'])],
 12: [('2019-03-02T06:35:48', ['1', '2', '3', '4'])]}

dict_expo_NGC4535={                                          #AO
 1: [('2018-04-09T03:18:03', ['1', '2', '3', '4'])],
 2: [('2018-04-09T04:46:39', ['1', '2', '3', '4'])],
 3: [('2018-04-10T02:42:01', ['1', '2', '3', '4'])],
 4: [('2018-04-14T04:31:10', ['1', '2', '3', '4'])],
 5: [('2018-04-16T04:53:52', ['1', '2', '3', '4'])],
 6: [('2018-05-17T00:00:23', ['1', '2', '3', '4'])]}

dict_expo_NGC5068={
 1:  [('2018-05-14T02:48:05', ['1', '2', '3', '4']),
      ('2018-06-14T02:46:50', ['1', '2', '3'])],
 2:  [('2018-05-14T04:20:06', ['1', '3', '4'])],             #removed one bad exposure
 3:  [('2018-05-15T02:42:20', ['1', '2', '3', '4'])],
 4:  [('2018-05-20T02:58:19', ['1', '2', '3', '4'])],
 5:  [('2018-05-21T04:13:30', ['1', '2', '3', '4'])],
 6:  [('2018-06-15T02:09:06', ['1', '2', '3', '4'])],
 7:  [('2018-06-17T01:57:12', ['1', '2', '3', '4'])],
 8:  [('2018-07-10T23:50:45', ['1', '2', '3', '4'])],
 9:  [('2018-07-11T00:56:19', ['1', '2', '3', '4'])],
 10: [('2018-07-14T00:44:22', ['1', '2', '3', '4'])]}

dict_expo_NGC1566={                                          #AO
 1:  [('2018-12-14T03:12:39', ['1', '2', '3', '4'])],
 2:  [('2019-01-15T02:28:00', ['1', '2', '3', '4'])],
# 3:  [('2019-01-16T02:36:25', ['4'])],                  #this is the pointing with problematic sky
 3:  [('2020-12-10T04:30:27', ['1', '2', '3', '4'])], # New updated pointing made end of 2020 to replace problematic sky pointing
 4:  [('2019-01-25T00:53:23', ['1', '2', '3', '4'])],
 5:  [('2019-01-27T00:52:13', ['1', '2', '3', '4'])],
 6:  [('2019-01-27T02:02:45', ['1', '2', '3', '4'])],
 7:  [('2019-01-28T01:09:07', ['1', '2', '3', '4'])],
 30: [('2017-10-23T04:45:57', ['1', '2', '3', '4'])]}

dict_expo_NGC3351={
 1:  [('2019-02-10T04:59:15', ['1', '2', '3', '4'])],        #there is a tmask for exp 1 NOT TO BE USE
 2:  [('2019-02-10T06:03:50', ['1', '2', '3', '4'])],
 3:  [('2019-03-02T03:17:26', ['1', '2', '3', '4'])],
 4:  [('2019-03-02T04:16:25', ['1', '2', '3', '4'])],
 5:  [('2019-03-03T03:51:18', ['1', '2', '3', '4'])],
 6:  [('2019-03-03T05:02:01', ['1', '2', '3', '4'])],
 7:  [('2019-03-11T02:49:48', ['1', '2', '3', '4'])],
 8:  [('2019-03-12T02:42:02', ['1', '2', '3', '4'])],
 30: [('2016-03-30T00:04:22', ['1', '2', '3', '4']),
      ('2016-04-04T00:43:01', ['1', '2', '3', '4'])]}
#we set 'use_fixed_pixtables=False' in order to not use the trail mask

dict_expo_NGC1433={
 1:  [('2018-10-16T06:52:54', ['1', '2', '3', '4'])],
 2:  [('2019-10-05T06:48:42', ['1', '2', '3', '4'])],
 3:  [('2019-10-05T07:57:53', ['1', '2', '3', '4'])],
 4:  [('2019-10-06T06:02:47', ['1', '2', '3', '4'])],
 5:  [('2019-10-07T06:46:16', ['1', '2', '3']), 
      ('2019-10-07T07:59:06', ['1'])],
 6:  [('2019-11-02T04:33:56', ['1', '2', '3', '4'])],
 7:  [('2019-11-20T02:08:18', ['1', '2', '3']), 
      ('2019-11-20T03:09:00', ['1'])],
 8:  [('2019-11-21T02:11:14', ['1', '2', '3', '4'])],
 9:  [('2019-11-22T06:27:26', ['1', '2', '3', '4'])],
 10: [('2019-12-20T04:30:19', ['1', '2', '3', '4'])],
 11: [('2019-12-21T02:16:23', ['1', '2', '3', '4'])],
 12: [('2019-12-21T04:27:47', ['1', '2', '3', '4'])],
 13: [('2019-12-22T04:24:22', ['1', '2']),
      ('2019-12-22T05:05:18', ['1', '2'])],
 14: [('2019-12-23T03:48:53', ['1', '2', '3', '4'])],
 15: [('2019-12-30T03:38:48', ['1', '2', '3', '4'])]}

dict_expo_NGC1300={
 1:  [('2019-02-03T01:41:13', ['1', '2', '3', '4'])],
 2:  [('2019-08-29T09:19:34', ['1', '2'])],        #3 and 4 removed from this one
 3:  [('2019-09-25T07:57:43', ['1', '2', '3', '4'])],
 4:  [('2019-10-08T07:39:28', ['1', '2', '3', '4'])],
 5:  [('2019-12-02T04:41:28', ['1', '2', '3', '4'])],
 6:  [('2019-12-03T05:12:50', ['1', '2', '3', '4'])],
 7:  [('2019-12-21T00:55:54', ['1', '2', '3', '4'])],
 8:  [('2019-12-23T01:39:45', ['1', '2', '3', '4'])],
 9:  [('2019-12-22T01:34:49', ['1', '2', '3', '4'])],
 10: [('2019-12-22T02:42:19', ['1', '2', '3', '4'])],
 11: [('2020-01-16T01:26:09', ['1', '2', '3', '4'])],
 12: [('2020-01-16T02:41:23', ['1', '2', '3', '4'])]}

dict_expo_NGC1385={
 1: [('2019-10-06T08:06:01', ['1', '2', '3', '4'])],
 2: [('2019-12-31T03:56:25', ['1', '2']), 
     ('2019-12-31T04:55:57', ['1', '2'])],
 3: [('2020-01-20T01:12:41', ['1', '2', '3', '4']), 
     ('2020-01-20T02:12:43', ['1', '2', '3', '4'])],
 4: [('2020-12-05T02:17:13', ['1', '2', '3', '4'])],
 5: [('2020-01-21T01:14:06', ['1', '2', '3', '4'])]}


dict_expo_NGC4303={
 1: [('2019-05-10T03:10:00', ['1']), 
     ('2019-05-10T03:51:19', ['1', '2', '3'])],
 2: [('2019-05-27T23:39:52', ['1', '2', '3', '4'])],
 3: [('2019-06-29T23:25:58', ['1', '2', '3']),		# removed expo 4
     ('2019-06-30T00:26:58', ['1'])],
 4: [('2020-01-30T07:08:21', ['1', '2']), 
     ('2020-01-30T07:43:58', ['1', '2'])],
 5: [('2020-02-03T06:23:31', ['1', '2', '3', '4'])],
 6: [('2020-02-03T07:35:27', ['1', '2', '3', '4'])],
 7: [('2020-02-28T07:39:20', ['1']), 
     ('2020-02-28T08:05:55', ['1', '2', '3'])],
 8: [('2020-02-19T05:54:25', ['1', '2', '3', '4'])],
 9: [('2020-02-19T07:35:28', ['1', '2', '3', '4'])]}

dict_expo_NGC4321={
 1:  [('2019-04-28T02:38:38', ['1', '2', '3', '4'])],
 2:  [('2019-04-30T02:20:03', ['1', '2', '3', '4'])],
 3:  [('2019-05-01T01:06:01', ['1', '2', '3', '4'])],
 4:  [('2020-03-02T06:11:38', ['1', '2', '3', '4'])],
 5:  [('2020-03-03T06:06:28', ['1', '2', '3', '4'])],
 6:  [('2020-02-20T07:07:56', ['1', '2', '3']), 
      ('2020-02-20T08:03:06', ['1'])],
 7:  [('2020-03-18T05:09:33', ['1', '2', '3', '4'])],
 8:  [('2021-02-12T06:35:52', ['1', '2', '3', '4'])],
 9:  [('2020-03-22T04:56:36', ['1', '2', '3', '4'])],
 10: [('2020-03-23T04:43:09', ['1', '2', '3', '4'])],
 11: [('2020-03-24T04:28:48', ['1', '2', '3', '4'])]}

dict_expo_NGC7496={
 1: [('2019-06-09T08:31:41', ['1']), 
     ('2019-06-09T08:53:47', ['1', '2', '3'])],
 2: [('2019-07-04T08:15:45', ['1', '2']),
     ('2019-07-04T09:23:58', ['1', '2', '3'])],
 3: [('2019-08-25T06:43:38', ['1', '2', '3', '4'])]}


dict_filter_for_alignment={
    'NGC0628':'WFI',
    'NGC1087':'WFI',
    'NGC1365':'WFI',
    'NGC1512':'WFI',
    'NGC1672':'WFI',
    'NGC2835':'WFI',
    'NGC3627':'WFI',
    'IC5332' :'WFI',
    'NGC4254':'WFI',
    'NGC4535':'WFI',
    'NGC5068':'WFI',
    'NGC1433':'WFI',
    'NGC1300':'WFI',
    'NGC1385':'WFI',
    'NGC1566':'WFI',
    'NGC3351':'WFI',
    'NGC4303':'WFI',
    'NGC4321':'WFI',
    'NGC7496':'DUPONT'
}
dict_locals = locals()
