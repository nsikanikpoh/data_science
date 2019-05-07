

DO $$ 
DECLARE
   num_points    INTEGER := 0;
BEGIN 
   SELECT team_id, team_name

    FROM teams LEFT JOIN matches
    ON host_team = team_id OR guest_team = team_id;

    CASE
        WHEN host_team = team_id AND host_goals > guest_goals THEN num_point += 3
        WHEN guest_team = team_id AND host_goals < guest_goals THEN num_point -= 3
        WHEN host_team = team_id AND host_goals = guest_goals THEN num_point += 1
    END

    ORDER BY num_points  DESC, team_id ASC;

END $$;